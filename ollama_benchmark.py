#!/usr/bin/env python3
"""Benchmark locally installed Ollama models and export a JSON report."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib import request

OLLAMA_URL = "http://localhost:11434/api/generate"
CONTEXT_TESTS = [8192, 16384, 32768, 65536, 131072]
DEFAULT_CONTEXT_WINDOW = 4096
BENCHMARK_PROMPT = "Explain the CAP theorem in distributed systems."
BENCHMARK_NUM_PREDICT = 200


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def run_command(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


def invoke_ollama_generate(model: str, prompt: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options or {},
        }
    ).encode("utf-8")

    req = request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_gpu_snapshot(command_runner: Callable[[list[str]], str] = run_command) -> dict[str, int]:
    try:
        raw = command_runner(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
        )
        first_line = next((line for line in raw.splitlines() if line.strip()), "")
        util_str, mem_str = [part.strip() for part in first_line.split(",", 1)]
        return {"util": int(util_str), "mem": int(mem_str)}
    except Exception:
        return {"util": 0, "mem": 0}


def get_cpu_snapshot(command_runner: Callable[[list[str]], str] = run_command) -> float:
    # Windows first (wmic), then Linux load average as a rough fallback.
    try:
        output = command_runner(["wmic", "cpu", "get", "loadpercentage"])
        match = re.search(r"(\d+)", output)
        if match:
            return round(float(match.group(1)), 1)
    except Exception:
        pass

    try:
        with open("/proc/loadavg", "r", encoding="utf-8") as fp:
            load1 = float(fp.read().split()[0])
        return round(load1 * 100.0, 1)
    except Exception:
        return 0.0


def parse_modelfile_options(modelfile: str) -> list[str]:
    if not modelfile or not modelfile.strip():
        return []

    names = {
        match.group(1)
        for line in modelfile.splitlines()
        if (match := re.match(r"^\s*PARAMETER\s+([^\s]+)\s+", line))
    }
    return sorted(names)


def parse_model_parameters(parameters_text: str) -> dict[str, str]:
    if not parameters_text or not parameters_text.strip():
        return {}

    parsed: dict[str, str] = {}
    for line in parameters_text.splitlines():
        match = re.match(r"^\s*([^\s]+)\s+(.+?)\s*$", line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        parsed.setdefault(key, value)
    return parsed


def get_installed_models(command_runner: Callable[[list[str]], str] = run_command) -> list[dict[str, str]]:
    try:
        parsed = json.loads(command_runner(["ollama", "list", "--json"]))
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except Exception:
        log("'ollama list --json' unsupported, falling back to plain 'ollama list' output parsing")

    list_output = command_runner(["ollama", "list"])
    models: list[dict[str, str]] = []
    for line in list_output.splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("NAME "):
            continue
        match = re.match(r"^([^\s]+)\s+", trimmed)
        if match:
            models.append({"name": match.group(1)})

    if not models:
        raise RuntimeError("Unable to parse installed models from 'ollama list' output.")

    return models


def get_model_show_json(model: str, command_runner: Callable[[list[str]], str] = run_command) -> dict[str, Any]:
    output = command_runner(["ollama", "show", model, "--json"])
    return json.loads(output)


def get_model_info_value(model_info: dict[str, Any], preferred_keys: list[str], suffix: str | None = None) -> Any:
    for key in preferred_keys:
        if key in model_info:
            return model_info[key]
    if suffix:
        for key in model_info:
            if key.endswith(suffix):
                return model_info[key]
    return None


def get_model_metadata(model: str, command_runner: Callable[[list[str]], str] = run_command) -> dict[str, Any]:
    show = get_model_show_json(model, command_runner=command_runner)
    model_info = show.get("model_info") or {}
    details = show.get("details") or {}

    architecture = str(get_model_info_value(model_info, ["general.architecture"]) or "")

    param_count = get_model_info_value(model_info, ["general.parameter_count"])
    parameters = round(float(param_count) / 1e9, 3) if param_count else 0

    embedding_length = int(
        get_model_info_value(
            model_info,
            ["llama.embedding_length", "qwen2.embedding_length", "gemma.embedding_length"],
            suffix=".embedding_length",
        )
        or 0
    )

    context_length = int(
        get_model_info_value(
            model_info,
            ["llama.context_length", "qwen2.context_length", "gemma.context_length"],
            suffix=".context_length",
        )
        or 0
    )

    quantization = str(get_model_info_value(model_info, ["general.file_type"]) or "")

    if not architecture and details.get("family"):
        architecture = str(details["family"])

    if parameters == 0 and isinstance(details.get("parameter_size"), str):
        match = re.search(r"([0-9]+\.?[0-9]*)", details["parameter_size"])
        if match:
            parameters = float(match.group(1))

    if not quantization and details.get("quantization_level"):
        quantization = str(details["quantization_level"])

    capabilities = show.get("capabilities") or ["completion"]
    if not capabilities:
        capabilities = ["completion"]

    input_types = sorted(set(["text"] + (["image"] if "vision" in capabilities else [])))
    output_types = sorted(set(["text"] + (["vector"] if "embedding" in capabilities else [])))
    accepts_tools = "tools" in capabilities
    reasoning = bool(re.search(r"(reason|r1|thinking)", model, flags=re.IGNORECASE))

    options = parse_model_parameters(str(show.get("parameters") or ""))
    options_available = parse_modelfile_options(str(show.get("modelfile") or ""))

    return {
        "architecture": architecture,
        "parameters": parameters,
        "embeddingLength": embedding_length,
        "contextLength": context_length,
        "quantization": quantization,
        "capabilities": capabilities,
        "inputTypes": input_types,
        "outputTypes": output_types,
        "acceptsTools": accepts_tools,
        "reasoning": reasoning,
        "options": options,
        "optionsAvailable": options_available,
    }


def test_context_window(model: str, generator: Callable[..., dict[str, Any]] = invoke_ollama_generate) -> int:
    max_stable = 0
    for ctx in CONTEXT_TESTS:
        log(f"Testing context {ctx}")
        try:
            response = generator(model, "hello", {"num_ctx": ctx, "num_predict": 8})
            if response.get("response"):
                max_stable = ctx
                log("success")
        except Exception:
            log("failed")
            break

    return max_stable or DEFAULT_CONTEXT_WINDOW


def get_efficiency_profile(
    tokens_per_second: float,
    first_token_latency_sec: float,
    load_time_sec: float,
    parameters_b: float,
    gpu_memory_mb: int,
) -> dict[str, float]:
    per_b = round(tokens_per_second / parameters_b, 3) if parameters_b > 0 else 0
    per_gb_vram = round(tokens_per_second / (gpu_memory_mb / 1024.0), 3) if gpu_memory_mb > 0 else 0
    score = (
        round(tokens_per_second / (1 + first_token_latency_sec + (load_time_sec / 2)), 3)
        if tokens_per_second > 0
        else 0
    )
    return {
        "tokensPerSecondPerBParameter": per_b,
        "tokensPerSecondPerGBVram": per_gb_vram,
        "responseEfficiencyScore": score,
    }


def get_workload_recommendations(
    tokens_per_second: float,
    first_token_latency_sec: float,
    context_window: int,
    accepts_tools: bool,
    capabilities: list[str],
    reasoning: bool,
) -> list[str]:
    recommendations: list[str] = []

    if first_token_latency_sec <= 1.0 and tokens_per_second >= 25:
        recommendations.append("interactive chat and coding assistants")
    if context_window >= 65536:
        recommendations.append("long-context summarization and retrieval workflows")
    if reasoning:
        recommendations.append("multi-step reasoning tasks")
    if accepts_tools:
        recommendations.append("tool-augmented agents and function-calling workflows")
    if "vision" in capabilities:
        recommendations.append("multimodal image+text analysis")
    if not recommendations:
        recommendations.append("general-purpose text generation")

    return sorted(set(recommendations))


def get_optimal_runtime_settings(
    context_window: int,
    tokens_per_second: float,
    first_token_latency_sec: float,
    model_options: dict[str, Any],
) -> dict[str, Any]:
    max_tokens = int(context_window / 4)

    try:
        quality_temperature = float(model_options.get("temperature", 0.7))
    except Exception:
        quality_temperature = 0.7

    latency_profile = "balanced"
    if first_token_latency_sec > 1.5 or tokens_per_second < 15:
        latency_profile = "throughput-conservative"
    elif first_token_latency_sec < 0.5 and tokens_per_second > 35:
        latency_profile = "low-latency"

    return {
        "profile": latency_profile,
        "recommendedOptions": {
            "num_ctx": context_window,
            "num_predict": min(max_tokens, 1024),
            "temperature": quality_temperature,
        },
    }


def benchmark_generation(
    model: str,
    generator: Callable[..., dict[str, Any]] = invoke_ollama_generate,
    gpu_snapshotter: Callable[[], dict[str, int]] = get_gpu_snapshot,
    cpu_snapshotter: Callable[[], float] = get_cpu_snapshot,
) -> dict[str, Any]:
    log("Running performance benchmark")

    gpu_before = gpu_snapshotter()
    cpu_before = cpu_snapshotter()

    response = generator(model, BENCHMARK_PROMPT, {"num_predict": BENCHMARK_NUM_PREDICT})

    gpu_after = gpu_snapshotter()
    cpu_after = cpu_snapshotter()

    load_sec = float(response.get("load_duration", 0)) / 1e9
    prompt_eval_sec = float(response.get("prompt_eval_duration", 0)) / 1e9
    eval_sec = float(response.get("eval_duration", 0)) / 1e9
    eval_count = int(response.get("eval_count", 0))

    tps = round(eval_count / eval_sec, 2) if eval_sec > 0 and eval_count > 0 else 0
    first_token_latency_sec = round(load_sec + prompt_eval_sec, 3)

    return {
        "loadTime": round(load_sec, 3),
        "firstToken": first_token_latency_sec,
        "tokensPerSecond": tps,
        "generatedTokens": eval_count,
        "promptTokens": int(response.get("prompt_eval_count", 0)),
        "evalDurationSec": round(eval_sec, 3),
        "gpuMem": max(gpu_before.get("mem", 0), gpu_after.get("mem", 0)),
        "gpuUtil": max(gpu_before.get("util", 0), gpu_after.get("util", 0)),
        "cpuUtil": round(max(cpu_before, cpu_after), 1),
    }


def run_benchmark(model_id: str | None = None) -> list[dict[str, Any]]:
    log("Discovering installed models")
    all_models = get_installed_models()

    if model_id:
        selected_models = [m for m in all_models if m.get("name") == model_id]
        if not selected_models:
            raise RuntimeError(f"Requested model '{model_id}' was not found in installed models.")
        log(f"Single-model mode enabled for: {model_id}")
    else:
        selected_models = all_models

    log(f"{len(selected_models)} models queued for benchmarking")

    results: list[dict[str, Any]] = []
    for model_entry in selected_models:
        model = str(model_entry.get("name") or "")
        if not model.strip():
            continue

        log("")
        log(f"Benchmarking {model}")

        meta = get_model_metadata(model)
        context_window = test_context_window(model)
        if meta["contextLength"] > 0:
            context_window = min(context_window, meta["contextLength"])

        max_tokens = int(context_window / 4)
        timing = benchmark_generation(model)

        efficiency = get_efficiency_profile(
            tokens_per_second=timing["tokensPerSecond"],
            first_token_latency_sec=timing["firstToken"],
            load_time_sec=timing["loadTime"],
            parameters_b=meta["parameters"],
            gpu_memory_mb=timing["gpuMem"],
        )

        optimal_runtime = get_optimal_runtime_settings(
            context_window=context_window,
            tokens_per_second=timing["tokensPerSecond"],
            first_token_latency_sec=timing["firstToken"],
            model_options=meta["options"],
        )

        workload_recommendations = get_workload_recommendations(
            tokens_per_second=timing["tokensPerSecond"],
            first_token_latency_sec=timing["firstToken"],
            context_window=context_window,
            accepts_tools=meta["acceptsTools"],
            capabilities=meta["capabilities"],
            reasoning=meta["reasoning"],
        )

        results.append(
            {
                "id": model,
                "name": model,
                "family": meta["architecture"],
                "parameters": meta["parameters"],
                "architecture": meta["architecture"],
                "reasoning": meta["reasoning"],
                "input": meta["inputTypes"],
                "output": meta["outputTypes"],
                "capabilities": meta["capabilities"],
                "acceptsTools": meta["acceptsTools"],
                "contextWindow": context_window,
                "maxTokens": max_tokens,
                "embeddingLength": meta["embeddingLength"],
                "quantization": meta["quantization"],
                "api": "ollama",
                "optionsAvailable": meta["optionsAvailable"],
                "defaultOptions": meta["options"],
                "optimalRuntimeSettings": optimal_runtime,
                "recommendedUseCases": workload_recommendations,
                "loadTimeSec": timing["loadTime"],
                "firstTokenLatencySec": timing["firstToken"],
                "tokensPerSecond": timing["tokensPerSecond"],
                "generatedTokens": timing["generatedTokens"],
                "promptTokens": timing["promptTokens"],
                "evalDurationSec": timing["evalDurationSec"],
                "efficiency": efficiency,
                "gpuMemoryMB": timing["gpuMem"],
                "gpuUtilizationPercent": timing["gpuUtil"],
                "cpuUtilizationPercent": timing["cpuUtil"],
            }
        )

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark locally installed Ollama models")
    parser.add_argument("--model-id", default=None, help="Benchmark only this model ID")
    parser.add_argument("--output-path", default="model-benchmark.json", help="Output JSON path")
    args = parser.parse_args(argv)

    results = run_benchmark(model_id=args.model_id)

    output_path = Path(args.output_path)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    log("Benchmark complete")
    log(f"Report written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
