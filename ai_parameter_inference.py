#!/usr/bin/env python3
"""
ai_parameter_inference.py

Automatically infer PARAMETER_FIELDS for generate_metadata.py using an LLM.

Workflow
--------
1. Parse params.input.
2. Read optional C++ source files (main.cc, problem.hh).
3. Ask an LLM to infer semantic name, datatype, QUDT unit,
   quantity kind, token index, confidence, and explanation.
4. Return PARAMETER_FIELDS.
"""

from __future__ import annotations

import configparser
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from openai import OpenAI, OpenAIError


# ============================================================
# Data classes
# ============================================================

@dataclass
class ParameterCandidate:
    section: str
    key: str
    value: str
    tokens: list[str]
    #: Optional hint scraped from getParam<Type>("Section.Key") call sites in
    #: main.cc/problem.hh -- e.g. "assigned to `density_` (C++ type `Scalar`)".
    #: See scan_getparam_hints() / attach_cpp_hints(). Empty if none found.
    cpp_hint: str = ""


@dataclass
class MetricCandidate:
    """A solution/output metric discovered in main.cc (e.g. a JSON key that
    the simulation writes to its results/summary file)."""
    key: str
    context: str = ""


@dataclass
class ParameterMetadata:
    semantic_name: str
    ini: tuple[str, str]
    index: int
    datatype: str
    unit: str
    quantityKind: str | None
    explanation: str = ""
    confidence: float = 1.0


# ============================================================
# File helpers
# ============================================================

def read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


# ============================================================
# Parse DUNE params.input
# ============================================================

def parse_dune_ini(path: Path) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(inline_comment_prefixes=("#",))
    cp.optionxform = str
    cp.read(path)
    return cp


def discover_parameters(params_input: Path) -> list[ParameterCandidate]:
    cp = parse_dune_ini(params_input)
    return [
        ParameterCandidate(
            section=section,
            key=key,
            value=value,
            tokens=value.split(),
        )
        for section in cp.sections()
        for key, value in cp.items(section)
    ]


# ============================================================
# getParam<Type>("Section.Key") call-site scanning
#
# DUNE/DuMux code reads params.input entries via calls like:
#   density_ = getParam<Scalar>("Component.LiquidDensity");
#   const auto radii = getParam<std::vector<Scalar>>("Grid.Radial0");
# The C++ variable name and type are strong, code-grounded evidence of a
# parameter's physical meaning (and therefore its SI unit) -- much stronger
# than pattern-matching the INI key name alone. We scrape these out of
# main.cc/problem.hh and attach them to matching ParameterCandidates.
# ============================================================

GETPARAM_PATTERN = re.compile(
    r'(?:(\w+)\s*=\s*)?'                       # optional "varname = "
    r'getParam(?:FromGroup)?\s*<(.+?)>\s*\('    # getParam<Type>( / getParamFromGroup<Type>(
    r'\s*(?:"[^"]*"\s*,\s*)?'                   # optional leading group-name arg
    r'"([^"]+)"'                                # the "Section.Key" string
)


def scan_getparam_hints(*source_texts: str) -> dict[str, str]:
    """Scan one or more C++ source files for getParam<Type>("Section.Key")
    call sites and return a dict keyed by lowercased "section.key" ->
    human-readable hint string describing the variable name and C++ type.
    """
    hints: dict[str, str] = {}
    for text in source_texts:
        if not text:
            continue
        for var, cpp_type, ini_key in GETPARAM_PATTERN.findall(text):
            cpp_type = cpp_type.strip()
            if var:
                hint = f"assigned to variable `{var}` (C++ type `{cpp_type}`)"
            else:
                hint = f"read with C++ type `{cpp_type}`"
            hints[ini_key.lower()] = hint
    return hints


def attach_cpp_hints(candidates: list[ParameterCandidate], *source_texts: str) -> None:
    """Mutate `candidates` in place, filling in cpp_hint for any candidate
    whose "Section.Key" matches a getParam<>() call site found in the given
    source texts (typically main.cc and problem.hh).
    """
    hints = scan_getparam_hints(*source_texts)
    for c in candidates:
        hint = hints.get(f"{c.section}.{c.key}".lower())
        if hint:
            c.cpp_hint = hint


# ============================================================
# Benchmark-level description (Doxygen doc-comment) extraction
#
# problem.hh (and sometimes main.cc) usually opens its class/problem
# definition with a /*! \brief ... */ Doxygen block describing the physical
# scenario and often citing the source publication, e.g.:
#   /*!
#    * \brief Test problem for the (Navier-) Stokes model in a 3D channel
#    * Benchmark case from Turek, Schaefer et al (1996) ...
#    */
# That's high-value context for both parameter and metric inference (it
# tells the LLM *what benchmark this is*, not just what code surrounds each
# value), and doubles as a human-readable description for the benchmark
# graph node. We grab the longest such block as a heuristic for "the main
# one" (per-function doc-comments are usually much shorter).
# ============================================================

DOXYGEN_BLOCK_PATTERN = re.compile(r"/\*!(.*?)\*/", re.DOTALL)


def extract_benchmark_description(*source_texts: str) -> str:
    """Return the longest Doxygen-style /*! ... */ comment block found
    across the given source texts, with the leading '*' decoration and any
    leading \\brief tag stripped. Returns "" if none is found.
    """
    best = ""
    for text in source_texts:
        if not text:
            continue
        for block in DOXYGEN_BLOCK_PATTERN.findall(text):
            cleaned = re.sub(r"^[ \t]*\*[ \t]?", "", block, flags=re.MULTILINE).strip()
            cleaned = re.sub(r"^\\brief\s*", "", cleaned)
            if len(cleaned) > len(best):
                best = cleaned
    return best


# ============================================================
# Discover output/solution metrics from main.cc
# ============================================================

#: Matches JSON keys emitted like: out << "  \"l2_error_pressure_rel\": " << value;
METRIC_KEY_PATTERN = re.compile(r'\\"([A-Za-z0-9_]+)\\"\s*:')


def discover_metrics(main_cc: str, context_chars: int = 100) -> list[MetricCandidate]:
    """Find output metric keys in main.cc and grab a small snippet of
    surrounding code around each occurrence, so the LLM has some context
    (e.g. what's being computed/printed) to infer a sensible SI unit from.
    """
    candidates: list[MetricCandidate] = []
    seen: set[str] = set()
    for m in METRIC_KEY_PATTERN.finditer(main_cc):
        key = m.group(1)
        if key in seen:
            continue
        seen.add(key)
        start = max(0, m.start() - context_chars)
        end = min(len(main_cc), m.end() + context_chars)
        candidates.append(MetricCandidate(key=key, context=main_cc[start:end].strip()))
    return candidates


# ============================================================
# Default scenario-specific parameter selection
# ============================================================

#: INI sections (case-insensitive) that are scenario-specific by default,
#: i.e. the parameters that typically differ between benchmark cases.
DEFAULT_SCENARIO_SECTIONS: set[str] = {"domain", "grid", "problem"}


def default_scenario_candidates(
    candidates: list[ParameterCandidate],
    sections: set[str] = DEFAULT_SCENARIO_SECTIONS,
) -> list[ParameterCandidate]:
    """Return the subset of candidates whose parent section is scenario-specific.

    A parameter is treated as scenario-specific by default when its parent
    INI section (matched case-insensitively) is one of `sections`, e.g.
    [Domain], [Grid], or [Problem]. Falls back to *all* candidates if none
    match, so callers never end up with an empty default selection.
    """
    wanted = {s.lower() for s in sections}
    selected = [c for c in candidates if c.section.lower() in wanted]
    return selected or list(candidates)


# ============================================================
# Pretty printing
# ============================================================

def print_candidates(candidates: list[ParameterCandidate]) -> None:
    print("\nDiscovered parameters\n")
    print("-" * 80)
    for i, p in enumerate(candidates):
        print(f"{i:2d} [{p.section}] {p.key:<25}{p.value}")
        if p.cpp_hint:
            print(f"      ↳ {p.cpp_hint}")
    print("-" * 80)


def print_metadata(metadata: list[dict]) -> None:
    print()
    print("=" * 90)
    print("Suggested parameter metadata")
    print("=" * 90)
    for i, item in enumerate(metadata):
        print(f"[{i}]")
        print(f"semantic name : {item['semantic_name']}")
        print(f"ini           : {tuple(item['ini'])}")
        print(f"datatype      : {item['datatype']}")
        print(f"unit          : {item['unit']}")
        print(f"quantity kind : {item['quantityKind']}")
        print(f"index         : {item['index']}")
        print(f"confidence    : {item['confidence']:.2f}")
        if item["explanation"]:
            print(f"reason        : {item['explanation']}")
        print()


def print_metric_candidates(candidates: list[MetricCandidate]) -> None:
    print("\nDiscovered output metrics\n")
    print("-" * 80)
    for i, m in enumerate(candidates):
        print(f"{i:2d} {m.key}")
    print("-" * 80)


def print_metric_metadata(metadata: list[dict]) -> None:
    print()
    print("=" * 90)
    print("Suggested metric metadata (SI units)")
    print("=" * 90)
    for i, item in enumerate(metadata):
        print(f"[{i}]")
        print(f"key           : {item['key']}")
        print(f"semantic name : {item['semantic_name']}")
        print(f"datatype      : {item['datatype']}")
        print(f"unit          : {item['unit']}")
        print(f"quantity kind : {item['quantityKind']}")
        print(f"confidence    : {item['confidence']:.2f}")
        if item["explanation"]:
            print(f"reason        : {item['explanation']}")
        print()


# ============================================================
# Prompt construction
# ============================================================

SYSTEM_PROMPT = """\
You are an expert in Computational Fluid Dynamics, DuMux, OpenFOAM,
scientific metadata, QUDT units, and JSON-LD.

Your task is to infer semantic metadata from DUNE params.input files.

For every parameter infer:
  - semantic_name
  - datatype  (schema:Integer | schema:Float | schema:String)
  - unit      (QUDT unit identifier)
  - quantityKind (QUDT quantityKind URI, or null if unknown)
  - index     (token index within the value string)
  - confidence
  - explanation

Some parameter entries include a "cpp_hint" field: this is scraped directly
from the source code's getParam<Type>("Section.Key") call site (e.g. the
exact C++ variable name and type it is assigned to, such as `density_` of
type `Scalar`). Treat cpp_hint as strong, code-grounded evidence of the
parameter's physical meaning and prefer it over guessing from the INI key
name alone -- e.g. a variable named `density_`/`rho_` implies kg/m3, a
variable read as `omega_`/an angular velocity implies rad/s, `viscosity_`
(dynamic) implies Pa*s, `radius_`/`length_` implies m.

A benchmark description (a doc-comment from the source, if available) and
the full main.cc/problem.hh source are provided only as background context
to help you understand what the parameters mean -- e.g. what physical
scenario or published benchmark this is. They will typically mention many
more parameters, functions, and variables than the ones you were asked
about. Do NOT infer metadata for anything other than the exact parameters
listed under "params.input" below.

STRICT OUTPUT RULE: the "params.input" section below lists exactly the
parameters you must return metadata for -- one JSON object per entry, same
section/key pairs, nothing added and nothing omitted. Do not include
parameters you merely noticed in main.cc, problem.hh, or the benchmark
description; only the ones explicitly listed under "params.input" below.

Respond with a raw JSON array only: the first character of your response must
be '[' and the last character must be ']'. Do not wrap the JSON in markdown
code fences (no ``` of any kind). Do not include any prose, explanation, or
preamble before or after the JSON.
"""

PROMPT_TEMPLATE = """\
The benchmark contains these files.

=====================
benchmark description
=====================

{benchmark_description}

=====================
params.input -- infer metadata for EXACTLY these {n_items} parameter(s), no others
=====================

{parameter_json}

=====================
main.cc (background context only -- do not infer metadata for anything here
that isn't also listed under params.input above)
=====================

{main_cc}

=====================
problem.hh (background context only -- do not infer metadata for anything
here that isn't also listed under params.input above)
=====================

{problem_hh}

Infer semantic metadata for every parameter listed under params.input above
-- exactly {n_items} item(s), no more, no less.

Return JSON like:

[
  {{
    "semantic_name": "cells_radial",
    "ini": ["Grid", "Cells0"],
    "index": 0,
    "datatype": "schema:Integer",
    "unit": "unit:UNITLESS",
    "quantityKind": "[qudt.org](http://qudt.org/vocab/quantitykind/Count)",
    "confidence": 0.99,
    "explanation": "..."
  }}
]

Return ONLY the raw JSON array. No markdown code fences, no explanation, no
text before the opening '[' or after the closing ']'.
"""


def build_prompt(
    candidates: list[ParameterCandidate],
    main_cc: str,
    problem_hh: str,
    benchmark_description: str = "",
) -> str:
    return PROMPT_TEMPLATE.format(
        benchmark_description=benchmark_description or "(none found)",
        n_items=len(candidates),
        parameter_json=json.dumps([asdict(c) for c in candidates], indent=2),
        main_cc=main_cc,
        problem_hh=problem_hh,
    )


# ------------------------------------------------------------
# Metric (output/solution quantity) prompts
# ------------------------------------------------------------

METRIC_SYSTEM_PROMPT = """\
You are an expert in Computational Fluid Dynamics, DuMux, OpenFOAM,
scientific metadata, QUDT units, and JSON-LD.

Your task is to infer semantic metadata for SOLUTION METRICS: output
quantities a CFD simulation writes to a results/summary file, as opposed to
input parameters.

For every metric infer:
  - semantic_name
  - datatype     (schema:Integer | schema:Float | schema:Double | schema:String)
  - unit         (QUDT unit identifier, using SI units for dimensional
                   quantities, e.g. unit:PA for pressure, unit:M-PER-SEC for
                   velocity, unit:SEC for time, unit:N for force)
  - quantityKind (QUDT quantityKind URI, or null if unknown)
  - confidence
  - explanation

Rules for choosing units:
  - Always prefer SI base or coherent derived units (pascal, metre, second,
    kilogram, metre per second, etc.) over any non-SI alternative.
  - If a metric name or its surrounding code indicates it is a relative,
    normalized, or ratio quantity (e.g. contains "rel", "relative",
    "normalized", "ratio"), treat it as dimensionless: use unit:UNITLESS and
    quantityKind http://qudt.org/vocab/quantitykind/DimensionlessRatio.
  - If a metric is an absolute error/norm of a dimensional field quantity
    (e.g. "l2_error_pressure_abs"), give it that field's own SI unit (e.g.
    pascal for pressure errors), not a unitless placeholder.
  - Use the benchmark description and surrounding main.cc/problem.hh code as
    context for what physical quantity the metric represents.

A benchmark description and the full main.cc/problem.hh source are provided
only as background context. They will typically mention many more metrics,
functions, and variables than the ones you were asked about.

STRICT OUTPUT RULE: the "metrics" section below lists exactly the metric
keys you must return metadata for -- one JSON object per key, nothing added
and nothing omitted. Do not include metrics you merely noticed in main.cc,
problem.hh, or the benchmark description; only the ones explicitly listed
under "metrics" below.

Respond with a raw JSON array only: the first character of your response must
be '[' and the last character must be ']'. Do not wrap the JSON in markdown
code fences (no ``` of any kind). Do not include any prose, explanation, or
preamble before or after the JSON.
"""

METRIC_PROMPT_TEMPLATE = """\
The benchmark writes these solution metrics to its results/summary JSON
file. Each entry below gives the metric's key plus a short snippet of the
surrounding main.cc code for context.

=====================
benchmark description
=====================

{benchmark_description}

=====================
metrics -- infer metadata for EXACTLY these {n_items} metric(s), no others
=====================

{metric_json}

=====================
main.cc (background context only -- do not infer metadata for anything here
that isn't also listed under metrics above)
=====================

{main_cc}

=====================
problem.hh (background context only -- do not infer metadata for anything
here that isn't also listed under metrics above)
=====================

{problem_hh}

Infer semantic metadata (with SI units) for every metric listed under
metrics above -- exactly {n_items} item(s), no more, no less.

Return JSON like:

[
  {{
    "key": "l2_error_pressure_abs",
    "semantic_name": "pressure_l2_error",
    "datatype": "schema:Double",
    "unit": "unit:PA",
    "quantityKind": "[qudt.org](http://qudt.org/vocab/quantitykind/Pressure)",
    "confidence": 0.95,
    "explanation": "..."
  }}
]

Return ONLY the raw JSON array. No markdown code fences, no explanation, no
text before the opening '[' or after the closing ']'.
"""


def build_metric_prompt(
    candidates: list[MetricCandidate],
    main_cc: str,
    problem_hh: str,
    benchmark_description: str = "",
) -> str:
    return METRIC_PROMPT_TEMPLATE.format(
        benchmark_description=benchmark_description or "(none found)",
        n_items=len(candidates),
        metric_json=json.dumps([asdict(c) for c in candidates], indent=2),
        main_cc=main_cc,
        problem_hh=problem_hh,
    )


# ============================================================
# Cache
# ============================================================

def cache_path(module_dir: Path) -> Path:
    return module_dir / ".parameter_metadata_cache.json"


def save_cache(module_dir: Path, metadata: list[dict]) -> None:
    cache_path(module_dir).write_text(json.dumps(metadata, indent=2))


def load_cache(module_dir: Path) -> list[dict] | None:
    path = cache_path(module_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def metric_cache_path(module_dir: Path) -> Path:
    return module_dir / ".metric_metadata_cache.json"


def save_metric_cache(module_dir: Path, metadata: list[dict]) -> None:
    metric_cache_path(module_dir).write_text(json.dumps(metadata, indent=2))


def load_metric_cache(module_dir: Path) -> list[dict] | None:
    path = metric_cache_path(module_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ============================================================
# Build PARAMETER_FIELDS from inferred metadata
# ============================================================

def build_parameter_fields(metadata: list[dict]) -> dict:
    """Keyed by the parameter's own (section, key) identity from
    params.input -- NOT by the LLM's semantic_name. Two different real
    parameters can never collide here even if the model (accidentally or
    not) assigns them the same semantic_name; semantic_name/explanation are
    carried along as descriptive metadata instead of used as the identity.
    """
    fields = {}
    for item in metadata:
        section, ini_key = item["ini"]
        stable_key = f"{section}::{ini_key}"
        fields[stable_key] = {
            "ini": (section, ini_key),
            "index": item.get("index", 0),
            "unit": item["unit"],
            "quantityKind": item.get("quantityKind"),
            "datatype": item["datatype"],
            "semantic_name": item["semantic_name"],
            "description": item.get("explanation", ""),
        }
    return fields


def build_metric_fields(metadata: list[dict]) -> dict:
    """Keyed by the raw metric key (as it appears in main.cc / the summary
    JSON file), since that's what generate_metadata.py's metric_keys list
    (and its downstream extract nodes) already use for lookups.
    """
    return {
        item["key"]: {
            "semantic_name": item["semantic_name"],
            "unit": item["unit"],
            "quantityKind": item.get("quantityKind"),
            "datatype": item["datatype"],
            "description": item.get("explanation", ""),
        }
        for item in metadata
    }


# ============================================================
# OpenAI client
# ============================================================

# ============================================================
# LLM provider configuration
# ============================================================

#: Groq and OpenAI both expose an OpenAI-compatible /v1/chat/completions
#: endpoint, so the same `openai` SDK works for either -- only the
#: base_url, API key, and default model differ.
PROVIDER_CONFIG: dict[str, dict[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "signup_url": "https://console.groq.com/keys",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "signup_url": "https://platform.openai.com/api-keys",
    },
}

DEFAULT_PROVIDER = "groq"


def get_client(provider: str = DEFAULT_PROVIDER) -> OpenAI:
    """
    Create an OpenAI-compatible client for the given provider.

    Requires the provider's API key env var to be set
    (GROQ_API_KEY for Groq, OPENAI_API_KEY for OpenAI).
    """
    if provider not in PROVIDER_CONFIG:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDER_CONFIG)}")

    config = PROVIDER_CONFIG[provider]
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"{config['api_key_env']} is not set in the environment. "
            f"Get a free key at {config['signup_url']}."
        )

    return OpenAI(api_key=api_key, base_url=config["base_url"])


# ============================================================
# Validation
# ============================================================

REQUIRED_FIELDS = ("semantic_name", "ini", "datatype", "unit")


def extract_json_array(raw: str) -> str:
    """Recover a JSON array from an LLM response that may be wrapped in
    markdown code fences or preceded/followed by stray prose.

    Some models (notably smaller/open-weight ones on Groq) ignore
    "no markdown fences" instructions and return things like:
        ```json
        [ ... ]
        ```
    or add a sentence before/after the array. This strips that wrapping
    so json.loads() gets clean input.
    """
    text = raw.strip()

    # Strip ```json ... ``` or ``` ... ``` fences.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    if text.startswith("[") and text.endswith("]"):
        return text

    # Fall back to slicing out the first top-level array in the text,
    # in case the model added a preamble or trailing remark.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return text


def validate_metadata(
    data: list[dict],
    candidates: list[ParameterCandidate] | None = None,
) -> list[dict]:
    """Ensure the returned JSON has the expected structure. If `candidates`
    is given, also enforce that the response contains *exactly* one item per
    requested candidate -- no more, no less. Models (especially smaller ones
    on Groq) sometimes ignore the requested subset and hallucinate extra
    entries for parameters they merely noticed elsewhere in the source code
    context; silently keeping those poisons the cache with metadata for
    parameters that may not even exist in every case's params.input, which
    later crashes resolve_case_params(). So any unrequested item is dropped
    (with a warning) rather than kept, and any candidate the model skipped
    raises an error so the caller retries instead of silently under-covering.
    """
    if not isinstance(data, list):
        raise ValueError("Model returned something other than a JSON list.")

    validated = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each item must be a JSON object.")

        for field in REQUIRED_FIELDS:
            if field not in item:
                raise ValueError(f"Missing required field '{field}'.")

        if not isinstance(item["ini"], list) or len(item["ini"]) != 2:
            raise ValueError("'ini' must be a list of [section, key].")

        item.setdefault("index", 0)
        item.setdefault("confidence", 1.0)
        item.setdefault("quantityKind", None)
        item.setdefault("explanation", "")
        validated.append(item)

    if candidates is not None:
        expected = {(c.section.lower(), c.key.lower()) for c in candidates}
        seen: set[tuple[str, str]] = set()
        filtered = []
        for item in validated:
            ini_pair = (item["ini"][0].lower(), item["ini"][1].lower())
            if ini_pair not in expected:
                print(
                    f"warning: dropping unrequested item '{item['semantic_name']}' "
                    f"({item['ini']}) -- not among the requested parameters",
                    file=sys.stderr,
                )
                continue
            if ini_pair in seen:
                print(f"warning: dropping duplicate item for {item['ini']}", file=sys.stderr)
                continue
            seen.add(ini_pair)
            filtered.append(item)

        missing = expected - seen
        if missing:
            missing_list = ", ".join(f"[{s}] {k}" for s, k in sorted(missing))
            raise ValueError(f"Model did not return metadata for: {missing_list}")

        validated = filtered

    return validated


REQUIRED_METRIC_FIELDS = ("key", "semantic_name", "datatype", "unit")


def validate_metric_metadata(
    data: list[dict],
    candidates: list[MetricCandidate] | None = None,
) -> list[dict]:
    """Ensure the returned metric JSON has the expected structure, and (if
    `candidates` is given) that it covers exactly the requested metric keys
    -- same rationale as validate_metadata() above.
    """
    if not isinstance(data, list):
        raise ValueError("Model returned something other than a JSON list.")

    validated = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each item must be a JSON object.")

        for field in REQUIRED_METRIC_FIELDS:
            if field not in item:
                raise ValueError(f"Missing required field '{field}'.")

        item.setdefault("confidence", 1.0)
        item.setdefault("quantityKind", None)
        item.setdefault("explanation", "")
        validated.append(item)

    if candidates is not None:
        expected = {c.key for c in candidates}
        seen: set[str] = set()
        filtered = []
        for item in validated:
            if item["key"] not in expected:
                print(
                    f"warning: dropping unrequested metric item '{item['key']}' "
                    f"-- not among the requested metrics",
                    file=sys.stderr,
                )
                continue
            if item["key"] in seen:
                print(f"warning: dropping duplicate item for metric '{item['key']}'", file=sys.stderr)
                continue
            seen.add(item["key"])
            filtered.append(item)

        missing = expected - seen
        if missing:
            raise ValueError(f"Model did not return metadata for metrics: {', '.join(sorted(missing))}")

        validated = filtered

    return validated


# ============================================================
# LLM query
# ============================================================

def _query_llm_json_array(
    *,
    provider: str,
    model: str | None,
    system_prompt: str,
    prompt: str,
    item_count: int,
    item_label: str,
    validator,
    retries: int,
    retry_delay: float,
    verbose: bool,
) -> list[dict]:
    """Shared request/retry/parse/validate loop used by both parameter and
    metric inference -- they only differ in prompts and validation rules.
    """
    client = get_client(provider)
    resolved_model = model or PROVIDER_CONFIG[provider]["default_model"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if verbose:
        print(f"\n[{provider}] endpoint   : {PROVIDER_CONFIG[provider]['base_url']}")
        print(f"[{provider}] model      : {resolved_model}")
        print(f"[{provider}] prompt size: {len(prompt):,} chars, {item_count} {item_label}(s)")

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            if verbose:
                print(f"\n--- Request (attempt {attempt}/{retries}) ---")
            start = time.monotonic()
            response = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=0,
            )
            elapsed = time.monotonic() - start
            raw_text = response.choices[0].message.content.strip()

            if verbose:
                usage = getattr(response, "usage", None)
                if usage is not None:
                    print(
                        f"--- Response (attempt {attempt}/{retries}, {elapsed:.2f}s, "
                        f"prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
                        f"total={usage.total_tokens} tokens) ---"
                    )
                else:
                    print(f"--- Response (attempt {attempt}/{retries}, {elapsed:.2f}s) ---")
                print(raw_text)
                print("--- end response ---\n")

            text = extract_json_array(raw_text)
            data = json.loads(text)
            return validator(data)

        except json.JSONDecodeError as exc:
            print(f"Attempt {attempt}/{retries}: JSON parse error — {exc}")
            last_error = exc

        except (OpenAIError, ValueError) as exc:
            print(f"Attempt {attempt}/{retries} failed — {exc}")
            last_error = exc

        if attempt < retries:
            time.sleep(retry_delay)

    raise RuntimeError(f"All {retries} attempts failed.") from last_error


def infer_parameter_metadata(
    candidates: list[ParameterCandidate],
    main_cc: str,
    problem_hh: str,
    *,
    benchmark_description: str = "",
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    retries: int = 3,
    retry_delay: float = 2.0,
    verbose: bool = False,
) -> list[dict]:
    """Ask an LLM (Groq or OpenAI) to infer semantic metadata for all discovered parameters."""
    prompt = build_prompt(candidates, main_cc, problem_hh, benchmark_description)
    return _query_llm_json_array(
        provider=provider,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        prompt=prompt,
        item_count=len(candidates),
        item_label="parameter",
        validator=lambda data: validate_metadata(data, candidates),
        retries=retries,
        retry_delay=retry_delay,
        verbose=verbose,
    )


def infer_metric_metadata(
    candidates: list[MetricCandidate],
    main_cc: str,
    problem_hh: str,
    *,
    benchmark_description: str = "",
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    retries: int = 3,
    retry_delay: float = 2.0,
    verbose: bool = False,
) -> list[dict]:
    """Ask an LLM (Groq or OpenAI) to infer semantic metadata -- with SI
    units -- for all discovered output/solution metrics.
    """
    prompt = build_metric_prompt(candidates, main_cc, problem_hh, benchmark_description)
    return _query_llm_json_array(
        provider=provider,
        model=model,
        system_prompt=METRIC_SYSTEM_PROMPT,
        prompt=prompt,
        item_count=len(candidates),
        item_label="metric",
        validator=lambda data: validate_metric_metadata(data, candidates),
        retries=retries,
        retry_delay=retry_delay,
        verbose=verbose,
    )


# ============================================================
# High-level entry point
# ============================================================

def run_inference(
    params_input: Path,
    main_cc_path: Path | None = None,
    problem_hh_path: Path | None = None,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    verbose: bool = False,
    include_metrics: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Parse input files, query the LLM, print and return inferred metadata
    for both input parameters and (if include_metrics and main.cc is given)
    output/solution metrics.

    Returns (parameter_metadata, metric_metadata) -- metric_metadata is []
    if include_metrics is False or no main.cc was provided.
    """
    candidates = discover_parameters(params_input)
    main_cc_text = read_text(main_cc_path)
    problem_hh_text = read_text(problem_hh_path)
    attach_cpp_hints(candidates, main_cc_text, problem_hh_text)
    benchmark_description = extract_benchmark_description(problem_hh_text, main_cc_text)
    print_candidates(candidates)

    parameter_metadata = infer_parameter_metadata(
        candidates=candidates,
        main_cc=main_cc_text,
        problem_hh=problem_hh_text,
        benchmark_description=benchmark_description,
        provider=provider,
        model=model,
        verbose=verbose,
    )
    print_metadata(parameter_metadata)

    metric_metadata: list[dict] = []
    if include_metrics and main_cc_text:
        metric_candidates = discover_metrics(main_cc_text)
        if metric_candidates:
            print_metric_candidates(metric_candidates)
            metric_metadata = infer_metric_metadata(
                candidates=metric_candidates,
                main_cc=main_cc_text,
                problem_hh=problem_hh_text,
                benchmark_description=benchmark_description,
                provider=provider,
                model=model,
                verbose=verbose,
            )
            print_metric_metadata(metric_metadata)

    return parameter_metadata, metric_metadata