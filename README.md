# Ollama Model Benchmark

A lightweight benchmarking tool for profiling locally installed Ollama models and exporting a structured report for model routing systems (for example OpenClaw).

The benchmark is designed to capture **real runtime behavior** on your own hardware so you can choose the right model for each workload.

---

## Project objective

This project benchmarks one or more installed Ollama models and reports:

- speed and responsiveness
- context capacity
- hardware usage
- efficiency indicators
- model capabilities and options
- practical workload recommendations

The output is intended to support automated model selection and runtime tuning.

---

## Requirements

- Windows with PowerShell
- Ollama running locally
- Optional: NVIDIA GPU + `nvidia-smi` for GPU metrics

If `nvidia-smi` is unavailable, GPU fields are reported as `0`.

---

## Usage

### Benchmark all installed models

```powershell
powershell -ExecutionPolicy Bypass -File .\ollama-benchmark.ps1
```

### Benchmark a single model (new)

```powershell
powershell -ExecutionPolicy Bypass -File .\ollama-benchmark.ps1 -ModelId "qwen2.5-coder:14b"
```

### Write to a custom output file

```powershell
powershell -ExecutionPolicy Bypass -File .\ollama-benchmark.ps1 -OutputPath ".\my-report.json"
```

---

## What the script now reports

For each model, the report includes:

- **Core metadata**
  - architecture
  - parameter count (B)
  - quantization
  - embedding length
- **Capabilities & IO**
  - capabilities discovered from `ollama show --json`
  - whether tools are accepted (`acceptsTools`)
  - supported input types (`text`, and `image` for vision models)
  - output types (`text`, plus `vector` for embedding models)
- **Runtime metrics**
  - load time
  - first token latency
  - generated token count
  - tokens/sec (based on Ollama token timing)
  - CPU/GPU utilization snapshots
- **Efficiency metrics**
  - tokens/sec per B parameters
  - tokens/sec per GB VRAM (when GPU metrics are available)
  - derived response efficiency score
- **Optimization guidance**
  - recommended runtime profile
  - suggested `num_ctx`, `num_predict`, and `temperature`
  - recommended use cases (interactive chat, long context, tools, vision, etc.)
- **Model runtime options**
  - options discovered from model parameter/modelfile data

---

## Benchmark process

1. Discover models with `ollama list --json`.
2. Read model metadata/capabilities via `ollama show --json`.
3. Probe stable context across configured context test sizes.
4. Run a generation benchmark with fixed prompt + token target.
5. Compute timing + efficiency + recommendation fields.
6. Write JSON report.

---

## Output file

Default output file:

```text
model-benchmark.json
```

Each entry is a complete model profile for routing and runtime decisions.

---

## Notes / limitations

- Context probing is practical, not exhaustive.
- Hardware snapshots are sampled around inference, not continuous traces.
- Some metadata fields depend on what each model exposes through Ollama.

---

## License

MIT License
