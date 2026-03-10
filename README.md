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
