#!/usr/bin/env python3
"""Collect and compare model metadata from Ollama and OpenRouter with hardware-aware recommendations."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_BENCHMARK_PATH = "model-benchmark.json"
DEFAULT_CATALOG_PATH = "model-comparison.json"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def run_command(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


def fetch_json(url: str, headers: dict[str, str] | None = None, timeout: int = 60) -> dict[str, Any]:
    req = request.Request(url, headers=headers or {}, method="GET")
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def detect_system_hardware(command_runner=run_command) -> dict[str, Any]:
    cpu_cores = os.cpu_count() or 0
    ram_gb = 0.0
    total_vram_gb = 0.0
    gpu_name = ""

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            meminfo = handle.read()
        match = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo)
        if match:
            ram_gb = round(int(match.group(1)) / (1024 * 1024), 2)
    except Exception:
        pass

    try:
        raw = command_runner(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
        )
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if lines:
            gpu_name = lines[0].split(",", 1)[0].strip()
            total_vram_mb = sum(int(line.split(",", 1)[1].strip()) for line in lines)
            total_vram_gb = round(total_vram_mb / 1024.0, 2)
    except Exception:
        pass

    return {
        "platform": platform.platform(),
        "cpu": {
            "architecture": platform.machine(),
            "cores": cpu_cores,
        },
        "memory": {
            "systemRamGB": ram_gb,
            "gpuVramGB": total_vram_gb,
        },
        "gpu": {
            "name": gpu_name,
            "detected": bool(gpu_name),
        },
    }


def parse_billions(value: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*([BM])", value, flags=re.IGNORECASE)
    if not match:
        return 0.0
    amount = float(match.group(1))
    unit = match.group(2).upper()
    if unit == "M":
        return round(amount / 1000.0, 3)
    return round(amount, 3)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def normalize_model_id(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    if ":latest" in raw:
        raw = raw.replace(":latest", "")
    return raw


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        if isinstance(value, (int, float)) and value == 0:
            continue
        return value
    return None


def read_json_file(path: str) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8-sig"))


def score_local_compatibility(required_gb: float, hardware: dict[str, Any]) -> int:
    if required_gb <= 0:
        return 3

    vram_gb = safe_float(hardware.get("memory", {}).get("gpuVramGB", 0))
    ram_gb = safe_float(hardware.get("memory", {}).get("systemRamGB", 0))

    if vram_gb > 0:
        if required_gb <= vram_gb * 0.85:
            return 1
        if required_gb <= vram_gb:
            return 2

    if ram_gb > 0:
        if required_gb <= ram_gb * 0.5:
            return 3
        if required_gb <= ram_gb:
            return 4

    return 5


def recommend_execution_mode(compatibility_score: int, has_api_pricing: bool) -> str:
    if compatibility_score <= 2:
        return "local-preferred"
    if compatibility_score >= 4 and has_api_pricing:
        return "api-preferred"
    if compatibility_score >= 4:
        return "not-recommended"
    return "hybrid"


def estimate_required_memory_gb(size_bytes: int, parameter_billions: float, quantization: str) -> float:
    if size_bytes > 0:
        return round((size_bytes / (1024**3)) * 1.2, 2)

    bits_lookup = {"q2": 2, "q3": 3, "q4": 4, "q5": 5, "q6": 6, "q8": 8, "fp16": 16, "bf16": 16}
    quant_key = str(quantization).lower()
    bits = 16
    for key, value in bits_lookup.items():
        if key in quant_key:
            bits = value
            break

    if parameter_billions > 0:
        bytes_required = parameter_billions * 1_000_000_000 * (bits / 8.0)
        return round((bytes_required / (1024**3)) * 1.15, 2)

    return 0.0


def ollama_to_entry(model: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
    details = model.get("details") or {}
    family = details.get("family") or "unknown"
    parameter_size = str(details.get("parameter_size") or "")
    params_b = parse_billions(parameter_size)
    quantization = str(details.get("quantization_level") or "")
    size_bytes = int(model.get("size") or 0)
    required_gb = estimate_required_memory_gb(size_bytes=size_bytes, parameter_billions=params_b, quantization=quantization)
    compatibility = score_local_compatibility(required_gb, hardware)

    return {
        "id": model.get("model") or model.get("name"),
        "name": model.get("name"),
        "family": family,
        "version": model.get("name"),
        "provider": "ollama",
        "contextLength": None,
        "parametersB": params_b,
        "quantization": quantization,
        "sizeGB": round(size_bytes / (1024**3), 3) if size_bytes else 0,
        "requiredMemoryGB": required_gb,
        "supportsToolUse": None,
        "inputModalities": ["text"],
        "outputModalities": ["text"],
        "pricing": {
            "inputPerMTokensUSD": 0.0,
            "outputPerMTokensUSD": 0.0,
            "notes": "Local Ollama execution",
        },
        "compatibilityScore": compatibility,
        "compatibilityLabel": compatibility_label(compatibility),
        "executionRecommendation": recommend_execution_mode(compatibility, has_api_pricing=False),
        "installed": True,
        "reasoning": None,
        "latencyP50Sec": None,
        "throughputP50TokensPerSec": None,
    }


def compatibility_label(score: int) -> str:
    labels = {
        1: "Excellent fit (fully compatible)",
        2: "Good fit (compatible)",
        3: "Usable with tradeoffs",
        4: "Poor fit (prefer API)",
        5: "Not compatible for local runtime",
    }
    return labels.get(score, "Unknown")


def openrouter_to_entry(model: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
    architecture = model.get("architecture") or {}
    pricing = model.get("pricing") or {}

    model_id = str(model.get("id") or "")
    family = model_id.split("/", 1)[0] if "/" in model_id else (architecture.get("tokenizer") or "unknown")
    params_b = parse_billions(str(model.get("name") or "") + " " + str(model.get("description") or ""))
    quantization = str(architecture.get("quantization") or "")
    required_gb = estimate_required_memory_gb(size_bytes=0, parameter_billions=params_b, quantization=quantization)
    compatibility = score_local_compatibility(required_gb, hardware)

    input_price = safe_float(pricing.get("prompt")) * 1_000_000
    output_price = safe_float(pricing.get("completion")) * 1_000_000

    return {
        "id": model_id,
        "name": model.get("name") or model_id,
        "family": family,
        "version": model_id.split("/", 1)[1] if "/" in model_id else model_id,
        "provider": "openrouter",
        "contextLength": model.get("context_length"),
        "parametersB": params_b,
        "quantization": quantization,
        "sizeGB": 0.0,
        "requiredMemoryGB": required_gb,
        "supportsToolUse": model.get("supported_parameters") and "tools" in model.get("supported_parameters", []),
        "inputModalities": architecture.get("input_modalities") or ["text"],
        "outputModalities": architecture.get("output_modalities") or ["text"],
        "pricing": {
            "inputPerMTokensUSD": round(input_price, 6),
            "outputPerMTokensUSD": round(output_price, 6),
            "notes": "OpenRouter API pricing",
        },
        "compatibilityScore": compatibility,
        "compatibilityLabel": compatibility_label(compatibility),
        "executionRecommendation": recommend_execution_mode(compatibility, has_api_pricing=True),
        "installed": False,
        "reasoning": None,
        "latencyP50Sec": None,
        "throughputP50TokensPerSec": None,
    }


def benchmark_to_entry(model: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
    params_b = safe_float(model.get("parameters"))
    quantization = str(model.get("quantization") or "")
    required_gb = estimate_required_memory_gb(size_bytes=0, parameter_billions=params_b, quantization=quantization)
    compatibility = score_local_compatibility(required_gb, hardware)
    model_id = str(model.get("id") or model.get("name") or "")
    return {
        "id": model_id,
        "name": model.get("name") or model_id,
        "family": str(model.get("family") or model.get("architecture") or "unknown"),
        "version": model_id,
        "provider": str(model.get("api") or "ollama"),
        "contextLength": model.get("contextWindow") or model.get("maxTokens"),
        "parametersB": params_b,
        "quantization": quantization,
        "sizeGB": 0.0,
        "requiredMemoryGB": required_gb,
        "supportsToolUse": "tools" in (model.get("capabilities") or []),
        "inputModalities": model.get("input") or ["text"],
        "outputModalities": ["text"],
        "pricing": {
            "inputPerMTokensUSD": 0.0,
            "outputPerMTokensUSD": 0.0,
            "notes": "Local benchmark metadata",
        },
        "compatibilityScore": compatibility,
        "compatibilityLabel": compatibility_label(compatibility),
        "executionRecommendation": recommend_execution_mode(compatibility, has_api_pricing=False),
        "installed": True,
        "reasoning": bool(model.get("reasoning")),
        "latencyP50Sec": safe_float(model.get("firstTokenLatencySec")) or None,
        "throughputP50TokensPerSec": safe_float(model.get("tokensPerSecond")) or None,
    }


def merge_entry(base: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    scalar_fields = [
        "name",
        "family",
        "version",
        "provider",
        "contextLength",
        "parametersB",
        "quantization",
        "sizeGB",
        "requiredMemoryGB",
        "supportsToolUse",
        "compatibilityScore",
        "compatibilityLabel",
        "executionRecommendation",
        "reasoning",
        "latencyP50Sec",
        "throughputP50TokensPerSec",
    ]
    for field in scalar_fields:
        preferred = first_non_empty(merged.get(field), fallback.get(field))
        if preferred is not None:
            merged[field] = preferred

    merged["inputModalities"] = first_non_empty(merged.get("inputModalities"), fallback.get("inputModalities")) or ["text"]
    merged["outputModalities"] = first_non_empty(merged.get("outputModalities"), fallback.get("outputModalities")) or ["text"]
    merged["installed"] = bool(merged.get("installed") or fallback.get("installed"))

    fallback_pricing = fallback.get("pricing") or {}
    merged_pricing = merged.get("pricing") or {}
    merged["pricing"] = {
        "inputPerMTokensUSD": first_non_empty(
            merged_pricing.get("inputPerMTokensUSD"), fallback_pricing.get("inputPerMTokensUSD")
        )
        or 0.0,
        "outputPerMTokensUSD": first_non_empty(
            merged_pricing.get("outputPerMTokensUSD"), fallback_pricing.get("outputPerMTokensUSD")
        )
        or 0.0,
        "notes": first_non_empty(merged_pricing.get("notes"), fallback_pricing.get("notes")) or "",
    }
    return merged


def build_hardware_recommendations(entries: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        entries,
        key=lambda item: (
            item.get("compatibilityScore", 5),
            not bool(item.get("installed")),
            item["pricing"].get("inputPerMTokensUSD", 0) + item["pricing"].get("outputPerMTokensUSD", 0),
        ),
    )
    best_local = [item["id"] for item in ranked if item.get("compatibilityScore", 5) <= 2][:5]
    best_installed = [item["id"] for item in ranked if item.get("installed") and item.get("compatibilityScore", 5) <= 3][:5]
    best_api = [item["id"] for item in ranked if item.get("provider") == "openrouter"][:5]
    return {
        "bestLocalCandidates": best_local,
        "bestInstalledLocal": best_installed,
        "bestApiCandidates": best_api,
    }


def build_family_summary(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        families.setdefault(str(entry.get("family") or "unknown").lower(), []).append(entry)

    summary: list[dict[str, Any]] = []
    for family_key in sorted(families):
        versions = families[family_key]
        best_local = min(versions, key=lambda item: (item.get("compatibilityScore", 5), item["pricing"]["inputPerMTokensUSD"]))
        cheapest_api = min(
            versions,
            key=lambda item: item["pricing"]["inputPerMTokensUSD"] + item["pricing"]["outputPerMTokensUSD"],
        )
        summary.append(
            {
                "family": family_key,
                "modelCount": len(versions),
                "providers": sorted({v["provider"] for v in versions}),
                "recommendedLocalModel": best_local["id"],
                "recommendedApiModel": cheapest_api["id"],
            }
        )
    return summary


def display_top_recommendations(entries: list[dict[str, Any]]) -> None:
    ranked = sorted(
        entries,
        key=lambda item: (
            item.get("compatibilityScore", 5),
            item["pricing"]["inputPerMTokensUSD"] + item["pricing"]["outputPerMTokensUSD"],
        ),
    )
    log("Top model recommendations")
    for item in ranked[:10]:
        total_cost = item["pricing"]["inputPerMTokensUSD"] + item["pricing"]["outputPerMTokensUSD"]
        print(
            f" - {item['id']:<40} | provider={item['provider']:<10} | compat={item['compatibilityScore']} | "
            f"cost_per_m={total_cost:.4f} | mode={item['executionRecommendation']}"
        )


def collect_data(openrouter_api_key: str | None = None) -> dict[str, Any]:
    hardware = detect_system_hardware()
    source_errors: list[str] = []

    log("Fetching Ollama model tags")
    try:
        ollama_payload = fetch_json(OLLAMA_TAGS_URL)
        ollama_models = ollama_payload.get("models") or []
    except Exception as exc:
        ollama_models = []
        source_errors.append(f"ollama: {exc}")

    log("Fetching OpenRouter model catalog")
    headers = {}
    if openrouter_api_key:
        headers["Authorization"] = f"Bearer {openrouter_api_key}"
    headers["HTTP-Referer"] = "https://local.model.compare"
    headers["X-Title"] = "Ollama Model Benchmark"

    try:
        openrouter_payload = fetch_json(OPENROUTER_MODELS_URL, headers=headers)
        openrouter_models = openrouter_payload.get("data") or []
    except Exception as exc:
        openrouter_models = []
        source_errors.append(f"openrouter: {exc}")

    benchmark_models = []
    try:
        benchmark_payload = read_json_file(DEFAULT_BENCHMARK_PATH)
        if isinstance(benchmark_payload, list):
            benchmark_models = benchmark_payload
    except Exception as exc:
        source_errors.append(f"model-benchmark.json: {exc}")

    catalog_models = []
    try:
        catalog_payload = read_json_file(DEFAULT_CATALOG_PATH)
        if isinstance(catalog_payload, dict):
            catalog_models = catalog_payload.get("models") or []
        elif isinstance(catalog_payload, list):
            catalog_models = catalog_payload
    except Exception as exc:
        source_errors.append(f"model-comparison.json: {exc}")

    entries = [ollama_to_entry(model, hardware) for model in ollama_models]
    entries.extend(openrouter_to_entry(model, hardware) for model in openrouter_models)
    entries.extend(benchmark_to_entry(model, hardware) for model in benchmark_models)

    for model in catalog_models:
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id") or model.get("name") or "")
        if model_id:
            entries.append(
                {
                    "id": model_id,
                    "name": model.get("name") or model_id,
                    "family": str(model.get("family") or "unknown"),
                    "version": model.get("version") or model_id,
                    "provider": model.get("provider") or "unknown",
                    "contextLength": model.get("contextLength"),
                    "parametersB": safe_float(model.get("parametersB")),
                    "quantization": str(model.get("quantization") or ""),
                    "sizeGB": safe_float(model.get("sizeGB")),
                    "requiredMemoryGB": safe_float(model.get("requiredMemoryGB")),
                    "supportsToolUse": model.get("supportsToolUse"),
                    "inputModalities": model.get("inputModalities") or ["text"],
                    "outputModalities": model.get("outputModalities") or ["text"],
                    "pricing": model.get("pricing") or {
                        "inputPerMTokensUSD": 0.0,
                        "outputPerMTokensUSD": 0.0,
                        "notes": "Catalog import",
                    },
                    "compatibilityScore": int(model.get("compatibilityScore") or 5),
                    "compatibilityLabel": model.get("compatibilityLabel") or compatibility_label(5),
                    "executionRecommendation": model.get("executionRecommendation") or "unknown",
                    "installed": bool(model.get("installed")),
                    "reasoning": model.get("reasoning"),
                    "latencyP50Sec": model.get("latencyP50Sec"),
                    "throughputP50TokensPerSec": model.get("throughputP50TokensPerSec"),
                }
            )

    merged_by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        norm_id = normalize_model_id(entry.get("id"))
        if not norm_id:
            continue
        if norm_id in merged_by_id:
            merged_by_id[norm_id] = merge_entry(merged_by_id[norm_id], entry)
        else:
            merged_by_id[norm_id] = entry

    final_entries = sorted(
        merged_by_id.values(),
        key=lambda item: (
            not bool(item.get("installed")),
            item.get("compatibilityScore", 5),
            str(item.get("name") or item.get("id") or "").lower(),
        ),
    )

    family_summary = build_family_summary(final_entries)
    hardware_recommendations = build_hardware_recommendations(final_entries)

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "ollama": OLLAMA_TAGS_URL,
            "openrouter": OPENROUTER_MODELS_URL,
        },
        "systemHardware": hardware,
        "sourceErrors": source_errors,
        "summary": {
            "totalModels": len(entries),
            "totalUniqueModels": len(final_entries),
            "ollamaModels": len(ollama_models),
            "openrouterModels": len(openrouter_models),
            "benchmarkModels": len(benchmark_models),
            "catalogModels": len(catalog_models),
            "families": len(family_summary),
        },
        "compatibilityScale": {
            "1": "Absolutely compatible (best local fit)",
            "2": "Compatible",
            "3": "Conditionally compatible",
            "4": "Likely incompatible for local runtime",
            "5": "Absolutely not compatible",
        },
        "familyRecommendations": family_summary,
        "hardwareRecommendations": hardware_recommendations,
        "models": final_entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Model comparison tool using Ollama + OpenRouter")
    parser.add_argument("--output", default="outputs/model_comparison.json", help="Output JSON file")
    parser.add_argument(
        "--openrouter-api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="Optional OpenRouter API key (or set OPENROUTER_API_KEY)",
    )
    args = parser.parse_args(argv)

    report = collect_data(openrouter_api_key=args.openrouter_api_key)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    display_top_recommendations(report["models"])
    log(f"Saved comparison report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
