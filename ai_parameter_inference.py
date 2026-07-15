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

Respond with a raw JSON array only: the first character of your response must
be '[' and the last character must be ']'. Do not wrap the JSON in markdown
code fences (no ``` of any kind). Do not include any prose, explanation, or
preamble before or after the JSON.
"""

PROMPT_TEMPLATE = """\
The benchmark contains these files.

=====================
params.input
=====================

{parameter_json}

=====================
main.cc
=====================

{main_cc}

=====================
problem.hh
=====================

{problem_hh}

Infer semantic metadata for every parameter.

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
) -> str:
    return PROMPT_TEMPLATE.format(
        parameter_json=json.dumps([asdict(c) for c in candidates], indent=2),
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


# ============================================================
# Build PARAMETER_FIELDS from inferred metadata
# ============================================================

def build_parameter_fields(metadata: list[dict]) -> dict:
    return {
        item["semantic_name"]: {
            "ini": tuple(item["ini"]),
            "index": item.get("index", 0),
            "unit": item["unit"],
            "quantityKind": item.get("quantityKind"),
            "datatype": item["datatype"],
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


def validate_metadata(data: list[dict]) -> list[dict]:
    """Ensure the returned JSON has the expected structure."""
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

    return validated


# ============================================================
# LLM query
# ============================================================

def infer_parameter_metadata(
    candidates: list[ParameterCandidate],
    main_cc: str,
    problem_hh: str,
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    retries: int = 3,
    retry_delay: float = 2.0,
    verbose: bool = False,
) -> list[dict]:
    """Ask an LLM (Groq or OpenAI) to infer semantic metadata for all discovered parameters."""
    client = get_client(provider)
    resolved_model = model or PROVIDER_CONFIG[provider]["default_model"]
    prompt = build_prompt(candidates, main_cc, problem_hh)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    if verbose:
        print(f"\n[{provider}] endpoint   : {PROVIDER_CONFIG[provider]['base_url']}")
        print(f"[{provider}] model      : {resolved_model}")
        print(f"[{provider}] prompt size: {len(prompt):,} chars, {len(candidates)} parameter(s)")

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
            return validate_metadata(data)

        except json.JSONDecodeError as exc:
            print(f"Attempt {attempt}/{retries}: JSON parse error — {exc}")
            last_error = exc

        except (OpenAIError, ValueError) as exc:
            print(f"Attempt {attempt}/{retries} failed — {exc}")
            last_error = exc

        if attempt < retries:
            time.sleep(retry_delay)

    raise RuntimeError(f"All {retries} attempts failed.") from last_error


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
) -> list[dict]:
    """Parse input files, query the LLM, print and return inferred metadata."""
    candidates = discover_parameters(params_input)
    print_candidates(candidates)

    metadata = infer_parameter_metadata(
        candidates=candidates,
        main_cc=read_text(main_cc_path),
        problem_hh=read_text(problem_hh_path),
        provider=provider,
        model=model,
        verbose=verbose,
    )

    print_metadata(metadata)
    return metadata