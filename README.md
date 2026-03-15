# Ollama Model Benchmark (Python)

A lightweight Python benchmarking tool for profiling locally installed Ollama models and exporting a structured report for model routing systems.

## Requirements

- Python 3.10+
- Ollama running locally
- Optional: NVIDIA GPU + `nvidia-smi` for GPU metrics

## Usage

Benchmark all installed models:

```bash
python ollama_benchmark.py
```

Benchmark a single model:

```bash
python ollama_benchmark.py --model-id "qwen2.5-coder:14b"
```

Write to a custom output file:

```bash
python ollama_benchmark.py --output-path ./my-report.json
```

By default, reports are written to the `outputs/` folder using timestamped names:

- Single model: `outputs/Ollama_<model>_benchmark_<timestamp>.json`
- Multi-model run: `outputs/Ollama_multi_benchmark_<timestamp>.json`

## Tests

Run the unit test suite:

```bash
python -m unittest discover -s tests -v
```

## Output

Default output location:

```text
outputs/
```

Each report entry includes metadata, capabilities, runtime performance, efficiency indicators, and workload recommendations.


## Model Comparison Tool (Ollama + OpenRouter)

Use `model_comparison_tool.py` to build a unified comparison JSON across local Ollama models and OpenRouter-hosted models, with hardware-aware compatibility scoring.

```bash
python model_comparison_tool.py --output outputs/model_comparison.json
```

Optional API key (also supported via `OPENROUTER_API_KEY`):

```bash
python model_comparison_tool.py --openrouter-api-key "$OPENROUTER_API_KEY"
```

The generated JSON includes:

- Local hardware scan (CPU, RAM, GPU/VRAM)
- Per-model details (family, version, params, quantization, context, tool support, modalities, cost)
- Compatibility score from **1 (best local fit)** to **5 (not locally compatible)**
- Execution recommendation (`local-preferred`, `hybrid`, `api-preferred`)
- Family-level local/API recommendations
