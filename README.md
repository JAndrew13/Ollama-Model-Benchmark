# Ollama Model Benchmark

A lightweight benchmarking tool for **profiling locally installed Ollama models** and generating a structured performance report for use in **agent routing systems** such as OpenClaw.

This project measures real-world model performance on your machine and exports a JSON file that can be used to automatically configure model selection logic for multi-model AI systems.

Instead of relying on generic model specs, the benchmark captures **actual runtime behavior** including latency, throughput, and hardware usage.

---

# Purpose

Local AI systems often run multiple models simultaneously. Each model has different:

* capabilities
* context limits
* inference speed
* hardware cost

Without empirical measurements it is difficult to determine which model should handle a request.

This tool solves that problem by automatically benchmarking every installed Ollama model and producing a **machine-specific routing profile**.

The output can be used to:

* configure OpenClaw model routing
* compare performance between models
* identify optimal models for different workloads
* analyze hardware utilization during inference

---

# Features

Automatic discovery of installed models.

Metadata extraction including:

* architecture
* parameter count
* embedding length
* quantization type

Runtime performance measurements:

* maximum stable context window
* recommended max tokens
* model load time
* first token latency
* tokens per second
* GPU memory usage
* GPU utilization
* CPU utilization

Outputs a **structured JSON benchmark report** suitable for automated configuration.

---

# Requirements

Windows system running:

* PowerShell
* Ollama
* NVIDIA GPU (optional but recommended)

Optional tools used for hardware metrics:

* `nvidia-smi` for GPU statistics

If `nvidia-smi` is not available the script will still run but GPU metrics will be reported as `0`.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_REPO/ollama-model-benchmark.git
```

Navigate to the project directory:

```bash
cd ollama-model-benchmark
```

Ensure Ollama is installed and running:

```bash
ollama list
```

---

# Usage

Run the benchmark script from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\ollama-benchmark.ps1
```

The script will automatically:

1. Discover installed Ollama models
2. Load each model
3. Probe context limits
4. Run a generation benchmark
5. Record hardware utilization
6. Export a JSON performance report

---

# Benchmark Process

For each model the script performs the following steps.

### 1. Model Discovery

Installed models are detected using:

```
ollama list
```

### 2. Metadata Extraction

Model metadata is extracted from:

```
ollama show <model>
```

Captured fields include:

* architecture
* parameter size
* embedding length
* quantization format

### 3. Context Window Test

The script incrementally tests context sizes:

```
8192
16384
32768
65536
131072
```

The largest successful context size is recorded as `contextWindow`.

### 4. Generation Benchmark

A short inference request is executed to measure:

* model load time
* first token latency
* generation throughput

### 5. Hardware Monitoring

System usage is sampled during inference:

GPU metrics via:

```
nvidia-smi
```

CPU usage via:

```
Get-Counter
```

---

# Output

The benchmark produces a file:

```
model-benchmark.json
```

Each entry contains both **model metadata and measured performance metrics**.

Example:

```json
{
  "id": "qwen2.5-coder:14b",
  "name": "qwen2.5-coder:14b",
  "family": "qwen2",
  "parameters": 14.8,
  "architecture": "qwen2",
  "reasoning": false,
  "input": ["text"],
  "capabilities": ["completion"],
  "contextWindow": 32768,
  "maxTokens": 8192,
  "embeddingLength": 5120,
  "quantization": "Q4_K_M",
  "api": "ollama",
  "loadTimeSec": 6.1,
  "firstTokenLatencySec": 0.78,
  "tokensPerSecond": 23.9,
  "gpuMemoryMB": 11342,
  "gpuUtilizationPercent": 91,
  "cpuUtilizationPercent": 18
}
```

---

# Interpreting Results

Three metrics are most useful for routing decisions.

**Load Time**

Time required to load the model into memory.

Large models may take 10–30 seconds to initialize.

**First Token Latency**

Time between sending the prompt and receiving the first token.

This determines perceived responsiveness.

**Tokens Per Second**

Generation throughput once the model begins producing text.

Higher values indicate faster long responses.

---

# Using the Benchmark with OpenClaw

The generated JSON can be imported into OpenClaw or other agent frameworks to drive routing decisions.

A router can estimate response cost using:

```
response_time ≈ load_time + first_token_latency + tokens / tokens_per_second
```

This allows the system to dynamically choose the fastest model capable of completing a task.

---

# Typical Performance Patterns

Local benchmarks often reveal patterns such as:

| Model Size | Load Time | Tokens/sec |
| ---------- | --------- | ---------- |
| 30B        | slow      | low        |
| 12–14B     | moderate  | high       |
| 7–8B       | very fast | moderate   |

In many systems mid-size models offer the best balance of latency and capability.

---

# Example Use Cases

Model routing for agent frameworks

Performance comparison across hardware

Evaluating quantization strategies

Selecting optimal models for production workloads

Hardware capacity planning for local inference systems

---

# Limitations

GPU metrics require NVIDIA hardware.

Instantaneous hardware sampling may miss short utilization spikes.

Context probing assumes models support extended context.

The benchmark prioritizes **relative comparison** rather than absolute performance measurement.

---

# Future Improvements

Potential enhancements include:

* peak VRAM detection
* continuous GPU sampling
* token-level timing
* automatic router configuration generation
* cross-machine benchmark comparison

---

# License

MIT License

---

# Acknowledgements

Built for experimentation with **local LLM orchestration systems** including OpenClaw and multi-model inference environments.
