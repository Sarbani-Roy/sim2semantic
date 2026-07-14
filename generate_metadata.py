#!/usr/bin/env python3
"""
generate_metadata.py

Generates a metadata4ing + Croissant JSON-LD semantic description ("RO-Crate"
style graph) for the rotating-cylinders benchmark directly from existing module artifacts.
"""
from __future__ import annotations

import argparse
import configparser
import json
import re
import sys
from pathlib import Path
from typing import Any

from ai_parameter_inference import (
    discover_parameters,
    infer_parameter_metadata,
    build_parameter_fields, 
    load_cache, 
    save_cache
)


# =============================================================================
# 1. JSON-LD @context (reused verbatim)
# =============================================================================

DEFAULT_CONTEXT = {
    "local": "https://github.com/Simulation-Benchmarks/rotating-cylinders/",
    "@vocab": "http://w3id.org/nfdi4ing/metadata4ing#",
    "dcterms": "http://purl.org/dc/terms/",
    "dcat": "http://www.w3.org/ns/dcat#",
    "schema": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "qudt": "http://qudt.org/schema/qudt/",
    "m4i": "http://w3id.org/nfdi4ing/metadata4ing#",
    "mathmod": "https://mardi4nfdi.de/mathmoddb#",
    "label": {"@id": "rdfs:label"},
    "Field": {"@id": "cr:Field"},
    "file object": {"@id": "cr:FileObject"},
    "method": {"@id": "m4i:Method"},
    "numerical variable": {"@id": "m4i:NumericalVariable"},
    "processing step": {"@id": "m4i:ProcessingStep"},
    "tool": {"@id": "m4i:Tool"},
    "has numerical value": {"@id": "m4i:hasNumericalValue"},
    "has unit": {"@id": "m4i:hasUnit"},
    "has quantity kind": {"@id": "m4i:hasKindOfQuantity"},
    "has part": {"@id": "obo:BFO_0000051"},
    "has input": {"@id": "obo:RO_0002233"},
    "has output": {"@id": "obo:RO_0002234"},
    "has employed tool": {"@id": "m4i:hasEmployedTool"},
    "has configuration": {"@id": "m4i:usesConfiguration"},
    "has parameter set": {"@id": "m4i:hasParameterSet"},
    "evaluates": {"@id": "m4i:evaluates"},
    "investigates": {"@id": "m4i:investigates"},
    "uses": "mathmod:uses",
    "describedAsDocumentedBy": "mathmod:describedAsDocumentedBy",
    "extract": {"@id": "cr:extract"},
    "jsonPath": {"@id": "cr:jsonPath"},
    "source": {"@id": "cr:source"},
    "represents": {"@id": "sio:SIO_000210"},
}

KNOWN_METRIC_UNITS: dict[str, dict[str, Any]] = {
    "l2_error_pressure_rel": {
        "unit": "unit:UNITLESS",
        "quantityKind": "http://qudt.org/vocab/quantitykind/Count",
    },
    "l2_error_velocity_rel": {
        "unit": "unit:UNITLESS",
        "quantityKind": "http://qudt.org/vocab/quantitykind/Count",
    },
}
DEFAULT_METRIC_UNIT: dict[str, Any] = {"unit": "unit:UNITLESS", "quantityKind": None}

DEFAULT_MANIFEST: dict[str, Any] = {
    "label": "rotating cylinders",
    "version": "1.0.0",
    "investigates_qid": "https://portal.mardi4nfdi.de/entity/Q6830614",
    "investigates_label": "Taylor\u2013Couette flow",
    "software_label": "DuMux",
    "publication_label": "Publication",
}


# =============================================================================
# 2. INI Parsing & Metric extraction
# =============================================================================

def parse_dune_ini(path: Path) -> dict[str, dict[str, str]]:
    cp = configparser.ConfigParser(inline_comment_prefixes=("#",))
    cp.optionxform = str
    text = path.read_text(encoding="utf-8")
    cp.read_string(text)
    return {section: dict(cp.items(section)) for section in cp.sections()}


def extract_token(ini: dict, section: str, key: str, index: int) -> str:
    if section not in ini or key not in ini[section]:
        raise KeyError(f"[{section}] {key} not found in params.input")
    tokens = ini[section][key].split()
    if not tokens:
        raise ValueError(f"[{section}] {key} has no value")
    return tokens[index]


def to_number(token: str) -> float | int | str:
    try:
        if re.fullmatch(r"[+-]?\d+", token):
            return int(token)
        return float(token)
    except ValueError:
        return token


METRIC_KEY_PATTERN = re.compile(r'\\"([A-Za-z0-9_]+)\\"\s*:')


def extract_metric_keys_from_maincc(main_cc_path: Path) -> list[str]:
    text = main_cc_path.read_text(encoding="utf-8")
    keys = list(dict.fromkeys(METRIC_KEY_PATTERN.findall(text)))
    if not keys:
        sys.exit(
            f"No JSON keys found in {main_cc_path} -- expected lines like "
            r'out << "  \"some_key\": " << value;'
        )
    return keys


# =============================================================================
# 3. Graph construction
# =============================================================================

def slugify(value: Any) -> str:
    s = str(value)
    s = s.replace(".", "p").replace("-", "m")
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


class GraphBuilder:
    def __init__(self, manifest: dict[str, Any], parameter_fields: dict[str, Any]):
        self.manifest = manifest
        self.parameter_fields = parameter_fields
        self.graph: list[dict[str, Any]] = []
        self._param_value_nodes: dict[tuple[str, Any], str] = {}
        self._extract_nodes: set[str] = set()
        self._metric_fields_built = False

    def _ensure_extract_node(self, key: str) -> str:
        extract_id = f"local:extract_{key}"
        if extract_id not in self._extract_nodes:
            self.graph.append({
                "@id": extract_id,
                "@type": "cr:DataSource",
                "jsonPath": f"/{key}",
            })
            self._extract_nodes.add(extract_id)
        return extract_id

    def add_parameter_variable(self, key: str, value: Any, spec: dict[str, Any]) -> str:
        json_key = spec.get("json_key", key)
        dedup_key = (key, value)
        if dedup_key in self._param_value_nodes:
            return self._param_value_nodes[dedup_key]

        suffix = f"{key}_{slugify(value)}"
        var_id = f"local:variable_{suffix}"
        field_id = f"local:field_{suffix}"
        source_id = f"local:source_{suffix}"
        extract_id = self._ensure_extract_node(json_key)

        var_node = {
            "@id": var_id,
            "label": key,
            "has numerical value": value,
            "has unit": {"@id": spec["unit"]},
        }
        if spec.get("quantityKind"):
            var_node["has quantity kind"] = {"@id": spec["quantityKind"]}

        self.graph += [
            var_node,
            {
                "@id": field_id, "@type": "Field",
                "dataType": {"@id": spec.get("datatype", "schema:Float")},
                "represents": {"@id": var_id},
                "source": {"@id": source_id},
            },
            {
                "@id": source_id, "@type": "cr:DataSource",
                "extract": {"@id": extract_id},
                "file object": {"@id": "local:parameter_file_object"},
            },
        ]
        self._param_value_nodes[dedup_key] = var_id
        return var_id

    def ensure_metric_fields(self, metric_keys: list[str]) -> list[str]:
        if self._metric_fields_built:
            raise RuntimeError("ensure_metric_fields() called twice")
        metric_ids = []
        for key in metric_keys:
            spec = KNOWN_METRIC_UNITS.get(key)
            if spec is None:
                print(f"warning: '{key}' (from main.cc) has no known unit annotation; "
                      f"defaulting to unitless.", file=sys.stderr)
                spec = DEFAULT_METRIC_UNIT
            slug = slugify(key)
            metric_id = f"local:metric_{slug}"
            field_id = f"local:field_{slug}"
            source_id = f"local:source_{slug}"
            extract_id = self._ensure_extract_node(key)
            self.graph += [
                {
                    "@id": metric_id, "@type": "numerical variable", "label": key,
                    "has unit": {"@id": spec["unit"]},
                    **({"has quantity kind": {"@id": spec["quantityKind"]}}
                       if spec.get("quantityKind") else {}),
                },
                {
                    "@id": field_id, "@type": "Field",
                    "dataType": {"@id": "schema:Double"},
                    "represents": {"@id": metric_id},
                    "source": {"@id": source_id},
                },
                {
                    "@id": source_id, "@type": "cr:DataSource",
                    "extract": {"@id": extract_id},
                    "file object": {"@id": "local:summary_file_object"},
                },
            ]
            metric_ids.append(metric_id)
        self._metric_fields_built = True
        return metric_ids

    def add_configuration(self, case_id: str, label: str, param_values: dict[str, Any]) -> str:
        var_ids = []
        for key, value in param_values.items():
            spec = self.parameter_fields[key]
            var_ids.append(self.add_parameter_variable(key, value, spec))
        config_id = f"local:configuration_{case_id}"
        self.graph.append({
            "@id": config_id,
            "@type": "m4i:ParameterSet",
            "label": label,
            "identifier": case_id,
            "has part": [{"@id": v} for v in var_ids],
        })
        return config_id

    def add_benchmark_node(self, config_ids: list[str], metric_ids: list[str]) -> None:
        m = self.manifest
        self.graph.insert(0, {
            "@id": "local:bm-rotating-cylinders",
            "@type": "m4i:Benchmark",
            "label": m["label"],
            "investigates": {"@id": m["investigates_qid"]},
            "uses": {"@id": m["investigates_qid"]},
            "evaluates": [{"@id": i} for i in metric_ids],
            "has parameter set": [{"@id": c} for c in config_ids],
            "describedAsDocumentedBy": {"@id": "local:publication"},
            "schema:version": m["version"],
        })
        self.graph.append({
            "@id": m["investigates_qid"],
            "@type": "mathmod:ResearchProblem",
            "label": m["investigates_label"],
        })
        self.graph.append({
            "@id": "local:publication",
            "@type": "mathmod:Publication",
            "label": m["publication_label"],
        })
        self.graph.append({
            "@id": "local:software", "@type": "tool", "label": m["software_label"],
        })
        self.graph.append({
            "@id": "local:parameter_file_object", "@type": "cr:FileObject",
            "label": "parameter.json",
        })
        self.graph.append({
            "@id": "local:summary_file_object", "@type": "cr:FileObject",
            "label": "solution_metrics.json",
        })


# =============================================================================
# 4. Case discovery & Resolution
# =============================================================================

def load_manifest(path: Path | None) -> dict[str, Any]:
    manifest = dict(DEFAULT_MANIFEST)
    if path is None:
        return manifest
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError:
            sys.exit("PyYAML not installed; use a .json manifest or `pip install pyyaml`")
        manifest.update(yaml.safe_load(text) or {})
    else:
        manifest.update(json.loads(text))
    return manifest


def resolve_case_params(case_dir: Path, parameter_fields: dict[str, Any]) -> dict[str, Any]:
    ini = parse_dune_ini(case_dir / "params.input")
    values: dict[str, Any] = {}
    for key, spec in parameter_fields.items():
        section, ini_key = spec["ini"]
        token = extract_token(ini, section, ini_key, spec["index"])
        values[key] = to_number(token)
    return values


def case_id_for(params_path: Path, root: Path) -> str:
    rel_dir = params_path.parent.relative_to(root)
    if str(rel_dir) == ".":
        return params_path.parent.name
    return "_".join(rel_dir.parts)


def discover_cases(root: Path) -> list[tuple[Path, str]]:
    found = sorted(root.rglob("params.input"))
    if not found:
        sys.exit(f"No params.input files found anywhere under {root}")

    cases: list[tuple[Path, str]] = []
    seen_ids: dict[str, Path] = {}
    for params_path in found:
        case_dir = params_path.parent
        case_id = case_id_for(params_path, root)
        if case_id in seen_ids:
            sys.exit(
                f"Duplicate case id '{case_id}' derived from two different directories:\n"
                f"  {seen_ids[case_id]}\n  {case_dir}"
            )
        seen_ids[case_id] = case_dir
        cases.append((case_dir, case_id))
    return cases


def find_main_cc(module_dir: Path, override: Path | None) -> Path:
    if override is not None:
        if not override.exists():
            sys.exit(f"--main-cc {override} does not exist")
        return override

    found = sorted(module_dir.rglob("main.cc"))
    if not found:
        sys.exit(f"No main.cc found under {module_dir}.")
    if len(found) > 1:
        listing = "\n  ".join(str(p) for p in found)
        sys.exit(f"Found multiple main.cc under {module_dir}, pass --main-cc to pick one:\n  {listing}")
    return found[0]


# =============================================================================
# 5. Execution Orchestration
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("module_dir", type=Path,
                     help="Path to the module folder containing both params.input file(s) and main.cc.")
    ap.add_argument("--main-cc", type=Path, default=None, dest="main_cc",
                     help="Explicit path to main.cc, only needed if module_dir contains more than one.")
    ap.add_argument("--manifest", type=Path, default=None, help="Optional benchmark_manifest.(yaml|json)")
    ap.add_argument("--output", type=Path, default=Path("metadata.jsonld"))
    ap.add_argument("--scenario-params", type=str, default=None,
                     help="Comma-separated list of raw parameter keys (e.g., 'Cells0,Cells1') that are scenario-specific. "
                          "If omitted, you will be prompted interactively.")
    args = ap.parse_args()

    if not args.module_dir.is_dir():
        sys.exit(f"Error: {args.module_dir} is not a directory")

    # 1. Path Resolution
    main_cc_path = find_main_cc(args.module_dir, args.main_cc)
    params_input_path = args.module_dir / "params.input"
    problem_hh_path = args.module_dir / "problem.hh"

    if not params_input_path.exists():
        discovered = sorted(args.module_dir.rglob("params.input"))
        if discovered:
            params_input_path = discovered[0]
        else:
            sys.exit(f"Error: Could not find a template params.input under {args.module_dir}")

    # 2. Discover Raw Parameters from INI file first (Before LLM)
    raw_candidates = discover_parameters(params_input_path)
    if not raw_candidates:
        sys.exit("Error: No raw parameters found in the template params.input file.")

    # 3. Filter raw candidates to determine which are Scenario-Specific
    selected_candidates = []
    raw_map = {c.key.lower(): c for c in raw_candidates}

    if args.scenario_params:
        provided_keys = [k.strip().lower() for k in args.scenario_params.split(",") if k.strip()]
        for pk in provided_keys:
            if pk in raw_map:
                selected_candidates.append(raw_map[pk])
            else:
                print(f"Warning: Param '{pk}' provided in --scenario-params was not found in params.input. Skipping.", file=sys.stderr)
        if not selected_candidates:
            print("Warning: None of the parameters specified in --scenario-params matched. Falling back to interactive selection.", file=sys.stderr)

    if not selected_candidates:
        # Prompt user interactively if in an interactive shell
        if not sys.stdin.isatty():
            print("\n[Non-interactive terminal detected]: Defaulting to treating ALL parameters as scenario-specific.")
            selected_candidates = list(raw_candidates)
        else:
            while True:
                print("\n=== Parameter Selection ===")
                print("Select parameters by typing their index numbers (e.g., 0, 2, 4).\n")
                
                # Print tabular structure (max 2 columns)
                cols = 2
                longest_formatted_len = max(len(f"[{i:2d}] {c.key}") for i, c in enumerate(raw_candidates))
                col_width = longest_formatted_len + 4  # padding margin
                
                for r_idx in range(0, len(raw_candidates), cols):
                    chunk = raw_candidates[r_idx:r_idx+cols]
                    row_str = "".join(f"[{r_idx + idx:2d}] {cand.key}".ljust(col_width) for idx, cand in enumerate(chunk))
                    print("  " + row_str)
                    
                print("\nInstructions:")
                print("  - Type the numbers of your selections separated by commas (e.g., 0, 2, 4)")
                print("  - Press Enter to select ALL parameters as scenario-specific")
                
                try:
                    user_input = input("\nSelect scenario-specific parameters: ").strip()
                    if user_input:
                        # Extract integers cleanly
                        indices = [int(i.strip()) for i in user_input.split(",") if i.strip().replace("-", "").isdigit()]
                        temp_selections = [raw_candidates[i] for i in indices if 0 <= i < len(raw_candidates)]
                        if not temp_selections:
                            print("No valid indices were selected. Please try again.", file=sys.stderr)
                            continue
                    else:
                        temp_selections = list(raw_candidates)
                        
                    # Show confirmation/correction step
                    print("\n--- Review Your Selection ---")
                    print(f"Scenario-Specific: {', '.join(c.key for c in temp_selections)}")
                    print(f"Global/Constant:   {', '.join(c.key for c in raw_candidates if c not in temp_selections) or 'None'}")
                    
                    confirm = input("\nIs this correct? Confirm (y) or Correct/Change (n): ").strip().lower()
                    if confirm in ("y", "yes", ""):
                        selected_candidates = temp_selections
                        break
                    else:
                        print("\nLet's correct your selections.")
                        continue
                        
                except (EOFError, KeyboardInterrupt):
                    print("\nInput cancelled. Falling back to treating ALL parameters as scenario-specific.", file=sys.stderr)
                    selected_candidates = list(raw_candidates)
                    break

    print(f"\nFinal Selection Confirmed.")
    print(f"Scenario-Specific: {', '.join(c.key for c in selected_candidates)}")
    print(f"Global/Constant:   {', '.join(c.key for c in raw_candidates if c not in selected_candidates) or 'None'}\n")

    # 4. Cache Management & AI Inference (ONLY for the selected parameters)
    metadata_cache = load_cache(args.module_dir) or []
    cache_lookup = {(item["ini"][0], item["ini"][1]): item for item in metadata_cache}
    
    final_metadata = []
    missing_candidates = []
    
    for candidate in selected_candidates:
        lookup_key = (candidate.section, candidate.key)
        if lookup_key in cache_lookup:
            final_metadata.append(cache_lookup[lookup_key])
        else:
            missing_candidates.append(candidate)
            
    if missing_candidates:
        print(f"\n[API DISABLED] Generating local fallback metadata for {len(missing_candidates)} parameters...")
        new_inferred = []
        for candidate in missing_candidates:
            new_inferred.append({
                "semantic_name": candidate.key,
                "ini": [candidate.section, candidate.key],
                "index": 0,
                "datatype": "schema:Float",
                "unit": "unit:UNITLESS",
                "quantityKind": None,
                "confidence": 0.0,
                "explanation": "Fallback generated because the OpenAI API query was commented out."
            })
        final_metadata.extend(new_inferred)
    else:
        print("Using cached parameter metadata (loaded from local file, 0 API queries triggered).")

    filtered_parameter_fields = build_parameter_fields(final_metadata)

    # 5. Process Cases & Build Graph
    manifest = load_manifest(args.manifest)
    cases = discover_cases(args.module_dir)
    metric_keys = extract_metric_keys_from_maincc(main_cc_path)

    builder = GraphBuilder(manifest, filtered_parameter_fields)
    metric_ids = builder.ensure_metric_fields(metric_keys)

    config_ids = []
    for case_dir, case_id in cases:
        params = resolve_case_params(case_dir, filtered_parameter_fields)
        label = ", ".join(
            f"{v} cells {k.replace('cells_', '')}"
            for k, v in params.items() if k.startswith("cells_")
        ) or case_id
        config_ids.append(builder.add_configuration(case_id, label, params))

    builder.add_benchmark_node(config_ids, metric_ids)

    doc = {"@context": DEFAULT_CONTEXT, "@graph": builder.graph}
    args.output.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} ({len(builder.graph)} graph nodes, {len(cases)} cases)")


if __name__ == "__main__":
    main()