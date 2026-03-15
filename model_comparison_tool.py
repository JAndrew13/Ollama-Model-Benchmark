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

    entries = [ollama_to_entry(model, hardware) for model in ollama_models]
    entries.extend(openrouter_to_entry(model, hardware) for model in openrouter_models)

    family_summary = build_family_summary(entries)

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
            "ollamaModels": len(ollama_models),
            "openrouterModels": len(openrouter_models),
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
        "models": entries,
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
