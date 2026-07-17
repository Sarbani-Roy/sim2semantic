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
    DEFAULT_PROVIDER,
    DEFAULT_SCENARIO_SECTIONS,
    PROVIDER_CONFIG,
    default_scenario_candidates,
    discover_parameters,
    discover_metrics,
    attach_cpp_hints,
    extract_benchmark_description,
    infer_parameter_metadata,
    infer_metric_metadata,
    build_parameter_fields,
    build_metric_fields,
    cache_path,
    metric_cache_path,
    load_cache,
    load_metric_cache,
    read_text,
    save_cache,
    save_metric_cache,
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

#: Optional manual overrides. Metric units/quantityKinds are inferred by the
#: LLM by default (see infer_metric_metadata / build_metric_fields), but any
#: key listed here takes precedence over the LLM's answer -- useful for
#: pinning a metric down without depending on/re-querying the model.
KNOWN_METRIC_UNITS: dict[str, dict[str, Any]] = {}
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


def discover_metrics_from_maincc(main_cc_path: Path):
    """Wrapper around ai_parameter_inference.discover_metrics() that also
    exits with a helpful error if no metric keys are found, and reads the
    file for the caller.
    """
    text = main_cc_path.read_text(encoding="utf-8")
    candidates = discover_metrics(text)
    if not candidates:
        sys.exit(
            f"No JSON keys found in {main_cc_path} -- expected lines like "
            r'out << "  \"some_key\": " << value;'
        )
    return candidates


# =============================================================================
# 3. Graph construction
# =============================================================================

def slugify(value: Any) -> str:
    s = str(value)
    s = s.replace(".", "p").replace("-", "m")
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


class GraphBuilder:
    def __init__(
        self,
        manifest: dict[str, Any],
        parameter_fields: dict[str, Any],
        metric_fields: dict[str, Any] | None = None,
        benchmark_description: str = "",
    ):
        self.manifest = manifest
        self.parameter_fields = parameter_fields
        #: LLM-inferred (or cached) metric metadata, keyed by raw metric key
        #: as it appears in main.cc / the summary JSON. See build_metric_fields().
        self.metric_fields = metric_fields or {}
        #: Doc-comment (e.g. Doxygen \brief + citation) scraped from
        #: problem.hh/main.cc -- see extract_benchmark_description(). Used as
        #: a human-readable description on the top-level benchmark node.
        self.benchmark_description = benchmark_description
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

    def add_parameter_variable(self, semantic_name: str, value: Any, spec: dict[str, Any]) -> str:
        # @id/label are tied to the actual params.input identity (section +
        # key) rather than the LLM-invented semantic_name, so they stay
        # stable and traceable back to the source file regardless of what
        # the model chose to call the parameter. semantic_name and the
        # LLM's explanation become a human-readable description instead.
        section, ini_key = spec["ini"]
        raw_label = f"{section}.{ini_key}"
        json_key = spec.get("json_key", ini_key)
        dedup_key = (raw_label, value)
        if dedup_key in self._param_value_nodes:
            return self._param_value_nodes[dedup_key]

        suffix = f"{slugify(section)}_{slugify(ini_key)}_{slugify(value)}"
        var_id = f"local:variable_{suffix}"
        field_id = f"local:field_{suffix}"
        source_id = f"local:source_{suffix}"
        extract_id = self._ensure_extract_node(json_key)

        description = spec.get("description") or ""
        if semantic_name and semantic_name != raw_label:
            description = f"{description} (inferred semantic name: {semantic_name})".strip()

        var_node = {
            "@id": var_id,
            "label": raw_label,
            "dcterms:description": description,
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
            # Precedence: manual override (KNOWN_METRIC_UNITS) > LLM-inferred
            # (self.metric_fields, built via build_metric_fields()) > unitless
            # fallback, in case a key somehow wasn't inferred.
            spec = KNOWN_METRIC_UNITS.get(key) or self.metric_fields.get(key)
            if spec is None:
                print(f"warning: '{key}' (from main.cc) has no known or "
                      f"inferred unit annotation; defaulting to unitless.", file=sys.stderr)
                spec = DEFAULT_METRIC_UNIT
            # @id/label stay tied to the raw metric key exactly as it appears
            # in main.cc / the summary JSON file; the LLM's semantic_name and
            # explanation become a description instead.
            semantic_name = spec.get("semantic_name", key)
            description = spec.get("description") or ""
            if semantic_name and semantic_name != key:
                description = f"{description} (inferred semantic name: {semantic_name})".strip()
            slug = slugify(key)
            metric_id = f"local:metric_{slug}"
            field_id = f"local:field_{slug}"
            source_id = f"local:source_{slug}"
            extract_id = self._ensure_extract_node(key)
            self.graph += [
                {
                    "@id": metric_id, "@type": "numerical variable", "label": key,
                    "dcterms:description": description,
                    "has unit": {"@id": spec["unit"]},
                    **({"has quantity kind": {"@id": spec["quantityKind"]}}
                       if spec.get("quantityKind") else {}),
                },
                {
                    "@id": field_id, "@type": "Field",
                    "dataType": {"@id": spec.get("datatype", "schema:Double")},
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
            var_ids.append(self.add_parameter_variable(spec.get("semantic_name", key), value, spec))
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
            **({"dcterms:description": self.benchmark_description} if self.benchmark_description else {}),
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
    ap.add_argument("--provider", type=str, choices=sorted(PROVIDER_CONFIG), default=DEFAULT_PROVIDER,
                     help=f"LLM provider used to infer parameter metadata (default: {DEFAULT_PROVIDER}). "
                          "Groq is free and requires GROQ_API_KEY; OpenAI requires OPENAI_API_KEY and billing.")
    ap.add_argument("--model", type=str, default=None,
                     help="Model name to use for the chosen --provider. Defaults to "
                          f"'{PROVIDER_CONFIG['groq']['default_model']}' for groq or "
                          f"'{PROVIDER_CONFIG['openai']['default_model']}' for openai.")
    ap.add_argument("--fallback-on-error", action="store_true",
                     help="If the LLM call fails (e.g. quota exceeded, network error), fall back to "
                          "local placeholder metadata (unitless, confidence 0.0) instead of exiting. "
                          "These placeholders are NOT written to the cache, so a future run will retry the API.")
    ap.add_argument("--verbose", action="store_true",
                     help="Print the request details and raw LLM response for each API call "
                          "(endpoint, model, token usage, timing, and the exact response text).")
    ap.add_argument("--clear-cache", action="store_true",
                     help="Delete any existing parameter/metric metadata cache for this module "
                          "before running, forcing fresh LLM inference for every selected item. "
                          "Without this flag, stale cache entries that no longer correspond to a "
                          "real params.input key or main.cc metric are pruned automatically, but "
                          "still-valid cached entries are reused as normal.")
    args = ap.parse_args()

    if not args.module_dir.is_dir():
        sys.exit(f"Error: {args.module_dir} is not a directory")

    if args.clear_cache:
        for p in (cache_path(args.module_dir), metric_cache_path(args.module_dir)):
            if p.exists():
                p.unlink()
                print(f"Cleared cache: {p}")

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

    # 2b. Enrich candidates with getParam<Type>("Section.Key") call-site hints
    #     scraped from main.cc/problem.hh -- e.g. "Component.LiquidDensity" ->
    #     assigned to `density_` (C++ type `Scalar`). This gives the LLM
    #     code-grounded evidence for picking the right SI unit, rather than
    #     just guessing from the INI key name.
    attach_cpp_hints(raw_candidates, read_text(main_cc_path), read_text(problem_hh_path))

    # Doc-comment (e.g. Doxygen \brief + literature citation) scraped from
    # problem.hh/main.cc -- passed to the LLM as high-level scenario context
    # for both parameter and metric inference, and recorded on the
    # benchmark graph node itself.
    benchmark_description = extract_benchmark_description(
        read_text(problem_hh_path), read_text(main_cc_path)
    )

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
        default_candidates = default_scenario_candidates(raw_candidates)
        default_indices = {
            i for i, c in enumerate(raw_candidates) if c in default_candidates
        }
        sections_label = ", ".join(sorted(DEFAULT_SCENARIO_SECTIONS))
        checked_indices = set(default_indices)

        while True:
            print("\n=== Parameter Selection ===")
            print(f"Parameters under [{sections_label}] are pre-selected by default (marked [x]).\n")

            # Print tabular structure (max 2 columns) with checkbox markers
            cols = 2
            longest_formatted_len = max(len(f"[x] {i:2d} {c.key}") for i, c in enumerate(raw_candidates))
            col_width = longest_formatted_len + 4  # padding margin

            for r_idx in range(0, len(raw_candidates), cols):
                chunk = raw_candidates[r_idx:r_idx + cols]
                row_str = "".join(
                    f"[{'x' if (r_idx + idx) in checked_indices else ' '}] {r_idx + idx:2d} {cand.key}".ljust(col_width)
                    for idx, cand in enumerate(chunk)
                )
                print("  " + row_str)

            print("\nInstructions:")
            print("  - Type index numbers to toggle them on/off (e.g., 0, 3, 5)")
            print("  - Type 'all' to select everything, or 'none' to clear the selection")
            print("  - Repeat as many times as needed -- each entry toggles from where you left off")
            print("  - Finally, press Enter with no input to confirm the selection and proceed")

            try:
                user_input = input("\nToggle selection (or Enter to confirm): ").strip().lower()

                if not user_input:
                    if not checked_indices:
                        print("No parameters selected. Please select at least one.", file=sys.stderr)
                        continue
                    selected_candidates = [c for i, c in enumerate(raw_candidates) if i in checked_indices]
                    break

                if user_input == "all":
                    checked_indices = set(range(len(raw_candidates)))
                    continue
                if user_input == "none":
                    checked_indices = set()
                    continue

                toggled = [int(tok.strip()) for tok in user_input.split(",") if tok.strip().lstrip("-").isdigit()]
                valid_toggled = [i for i in toggled if 0 <= i < len(raw_candidates)]
                if not valid_toggled:
                    print("No valid indices recognized. Please try again.", file=sys.stderr)
                    continue
                for i in valid_toggled:
                    checked_indices.symmetric_difference_update({i})

            except (EOFError, KeyboardInterrupt):
                print(
                    f"\nInput cancelled. Falling back to the default: parameters under [{sections_label}].",
                    file=sys.stderr,
                )
                selected_candidates = default_candidates
                break

    print(f"\nFinal Selection Confirmed.")
    print(f"Scenario-Specific: {', '.join(c.key for c in selected_candidates)}")
    print(f"Global/Constant:   {', '.join(c.key for c in raw_candidates if c not in selected_candidates) or 'None'}\n")

    # 4. Cache Management & AI Inference (ONLY for the selected parameters)
    metadata_cache = load_cache(args.module_dir) or []

    # Prune any cached entries that no longer correspond to a real [section]
    # key in the current params.input -- e.g. leftovers from a prior buggy
    # run where the LLM hallucinated extra parameters it merely noticed in
    # the source code. Left in place, those can silently resurface and crash
    # resolve_case_params() later, exactly as happened before this check
    # existed. Genuinely valid cached entries are kept and NOT re-queried.
    valid_param_keys = {(c.section, c.key) for c in raw_candidates}
    pruned_metadata_cache = [item for item in metadata_cache if tuple(item["ini"]) in valid_param_keys]
    if len(pruned_metadata_cache) != len(metadata_cache):
        dropped = len(metadata_cache) - len(pruned_metadata_cache)
        print(
            f"Pruned {dropped} stale parameter cache entr{'y' if dropped == 1 else 'ies'} "
            "no longer present in params.input.",
            file=sys.stderr,
        )
        save_cache(args.module_dir, pruned_metadata_cache)
    metadata_cache = pruned_metadata_cache

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
        resolved_model = args.model or PROVIDER_CONFIG[args.provider]["default_model"]
        print(f"\nQuerying {args.provider} ({resolved_model}) for {len(missing_candidates)} parameter(s) not found in cache...")
        try:
            new_inferred = infer_parameter_metadata(
                candidates=missing_candidates,
                main_cc=read_text(main_cc_path),
                problem_hh=read_text(problem_hh_path),
                benchmark_description=benchmark_description,
                provider=args.provider,
                model=args.model,
                verbose=args.verbose,
            )
        except (RuntimeError, ValueError) as exc:
            if not args.fallback_on_error:
                sys.exit(
                    f"Error: {args.provider} parameter inference failed -- {exc}\n"
                    "Pass --fallback-on-error to use local placeholder metadata instead of exiting."
                )

            print(
                f"Warning: {args.provider} inference failed ({exc}). "
                f"Using local placeholder metadata for {len(missing_candidates)} parameter(s) instead.",
                file=sys.stderr,
            )
            new_inferred = [
                {
                    "semantic_name": candidate.key,
                    "ini": [candidate.section, candidate.key],
                    "index": 0,
                    "datatype": "schema:Float",
                    "unit": "unit:UNITLESS",
                    "quantityKind": None,
                    "confidence": 0.0,
                    "explanation": "Placeholder generated because the OpenAI API call failed.",
                }
                for candidate in missing_candidates
            ]
            final_metadata.extend(new_inferred)
            # Deliberately NOT written to the cache -- placeholders are low-confidence
            # stand-ins, so a future run should retry the API rather than reuse them.
        else:
            final_metadata.extend(new_inferred)

            # Merge the newly inferred entries into the on-disk cache, keeping
            # any previously cached entries untouched.
            merged_cache = dict(cache_lookup)
            for item in new_inferred:
                merged_cache[(item["ini"][0], item["ini"][1])] = item
            save_cache(args.module_dir, list(merged_cache.values()))
    else:
        print("Using cached parameter metadata (loaded from local file, 0 API queries triggered).")

    filtered_parameter_fields = build_parameter_fields(final_metadata)

    # 4b. Cache Management & AI Inference for output/solution METRICS
    #     (same idea as step 4, but for the JSON keys main.cc writes out,
    #     inferred in SI units rather than picked from KNOWN_METRIC_UNITS.)
    metric_candidates = discover_metrics_from_maincc(main_cc_path)
    metric_keys = [c.key for c in metric_candidates]

    metric_cache = load_metric_cache(args.module_dir) or []

    # Same staleness pruning as for parameters above, but keyed on the raw
    # metric key against what's currently found in main.cc.
    valid_metric_keys = {c.key for c in metric_candidates}
    pruned_metric_cache = [item for item in metric_cache if item["key"] in valid_metric_keys]
    if len(pruned_metric_cache) != len(metric_cache):
        dropped = len(metric_cache) - len(pruned_metric_cache)
        print(
            f"Pruned {dropped} stale metric cache entr{'y' if dropped == 1 else 'ies'} "
            "no longer present in main.cc.",
            file=sys.stderr,
        )
        save_metric_cache(args.module_dir, pruned_metric_cache)
    metric_cache = pruned_metric_cache

    metric_cache_lookup = {item["key"]: item for item in metric_cache}

    final_metric_metadata = []
    missing_metric_candidates = []
    for candidate in metric_candidates:
        if candidate.key in metric_cache_lookup:
            final_metric_metadata.append(metric_cache_lookup[candidate.key])
        else:
            missing_metric_candidates.append(candidate)

    if missing_metric_candidates:
        resolved_model = args.model or PROVIDER_CONFIG[args.provider]["default_model"]
        print(f"\nQuerying {args.provider} ({resolved_model}) for "
              f"{len(missing_metric_candidates)} metric(s) not found in cache...")
        try:
            new_inferred_metrics = infer_metric_metadata(
                candidates=missing_metric_candidates,
                main_cc=read_text(main_cc_path),
                problem_hh=read_text(problem_hh_path),
                benchmark_description=benchmark_description,
                provider=args.provider,
                model=args.model,
                verbose=args.verbose,
            )
        except (RuntimeError, ValueError) as exc:
            if not args.fallback_on_error:
                sys.exit(
                    f"Error: {args.provider} metric inference failed -- {exc}\n"
                    "Pass --fallback-on-error to use local placeholder metadata instead of exiting."
                )
            print(
                f"Warning: {args.provider} metric inference failed ({exc}). "
                f"Using local placeholder metadata for {len(missing_metric_candidates)} metric(s) instead.",
                file=sys.stderr,
            )
            new_inferred_metrics = [
                {
                    "key": candidate.key,
                    "semantic_name": candidate.key,
                    "datatype": "schema:Double",
                    "unit": "unit:UNITLESS",
                    "quantityKind": None,
                    "confidence": 0.0,
                    "explanation": "Placeholder generated because the LLM API call failed.",
                }
                for candidate in missing_metric_candidates
            ]
            final_metric_metadata.extend(new_inferred_metrics)
            # Deliberately NOT written to the cache, same rationale as parameters.
        else:
            final_metric_metadata.extend(new_inferred_metrics)
            merged_metric_cache = dict(metric_cache_lookup)
            for item in new_inferred_metrics:
                merged_metric_cache[item["key"]] = item
            save_metric_cache(args.module_dir, list(merged_metric_cache.values()))
    else:
        print("Using cached metric metadata (loaded from local file, 0 API queries triggered).")

    filtered_metric_fields = build_metric_fields(final_metric_metadata)

    # 5. Process Cases & Build Graph
    manifest = load_manifest(args.manifest)
    cases = discover_cases(args.module_dir)

    builder = GraphBuilder(manifest, filtered_parameter_fields, filtered_metric_fields, benchmark_description)
    metric_ids = builder.ensure_metric_fields(metric_keys)

    config_ids = []
    for case_dir, case_id in cases:
        params = resolve_case_params(case_dir, filtered_parameter_fields)
        cell_parts = []
        for k, v in params.items():
            name = filtered_parameter_fields[k].get("semantic_name", "")
            if name.lower().startswith("cells_"):
                cell_parts.append(f"{v} cells {name[len('cells_'):]}")
        label = ", ".join(cell_parts) or case_id
        config_ids.append(builder.add_configuration(case_id, label, params))

    builder.add_benchmark_node(config_ids, metric_ids)

    doc = {"@context": DEFAULT_CONTEXT, "@graph": builder.graph}
    args.output.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} ({len(builder.graph)} graph nodes, {len(cases)} cases)")


if __name__ == "__main__":
    main()