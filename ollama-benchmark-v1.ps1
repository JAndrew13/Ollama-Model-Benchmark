$ErrorActionPreference = "Stop"

$OllamaUrl = "http://localhost:11434/api/generate"
$OutputFile = "model-benchmark.json"

$contextTests = @(8192,16384,32768,65536,131072)

function Log {
    param($msg)
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "[$ts] $msg"
}

function GetGPU {
    try {
        $raw = nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
        $parts = $raw.Split(",")
        return @{
            util = [int]$parts[0].Trim()
            mem  = [int]$parts[1].Trim()
        }
    }
    catch {
        return @{
            util = 0
            mem  = 0
        }
    }
}

function GetCPU {
    try {
        $cpu = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
        return [math]::Round($cpu,1)
    }
    catch {
        return 0
    }
}

function TestContext {
    param($model)

    $max = 0

    foreach ($ctx in $contextTests) {

        Log "Testing context $ctx"

        try {

            $body = @{
                model  = $model
                prompt = "hello"
                stream = $false
                options = @{
                    num_ctx = $ctx
                }
            } | ConvertTo-Json -Depth 5

            $resp = Invoke-RestMethod `
                -Uri $OllamaUrl `
                -Method Post `
                -Body $body `
                -ContentType "application/json"

            if ($resp.response) {
                Log "success"
                $max = $ctx
            }

        }
        catch {
            Log "failed"
            break
        }
    }

    return $max
}

function BenchmarkGeneration {
    param($model)

    Log "Running performance benchmark"

    $prompt = "Explain the CAP theorem in distributed systems."

    $body = @{
        model = $model
        prompt = $prompt
        stream = $false
        options = @{
            num_predict = 200
        }
    } | ConvertTo-Json -Depth 5

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    $resp = Invoke-RestMethod `
        -Uri $OllamaUrl `
        -Method Post `
        -Body $body `
        -ContentType "application/json"

    $sw.Stop()

    $gpu = GetGPU
    $cpu = GetCPU

    $time = $sw.Elapsed.TotalSeconds
    $tps = 200 / $time

    return @{
        loadTime = $resp.load_duration / 1000000000
        firstToken = $resp.prompt_eval_duration / 1000000000
        tokensPerSecond = [math]::Round($tps,2)
        gpuMem = $gpu.mem
        gpuUtil = $gpu.util
        cpuUtil = $cpu
    }
}

Log "Discovering installed models"

$models = ollama list | Select-Object -Skip 1 | ForEach-Object {
    ($_ -split "\s+")[0]
}

Log "$($models.Count) models detected"

$results = @()

foreach ($model in $models) {

    Log ""
    Log "Benchmarking $model"

    $show = ollama show $model

    $architecture = ""
    $parameters = 0
    $embedding = 0
    $quantization = ""

    foreach ($line in $show) {

        if ($line -like "*architecture*") {
            $architecture = ($line -split '\s+')[-1]
        }

if ($line -like "*parameters*") {

    $match = [regex]::Match($line, '([0-9]+(\.[0-9]+)?)B')

    if ($match.Success) {
        $parameters = [double]$match.Groups[1].Value
    }
}

if ($line -like "*embedding length*") {

    $match = [regex]::Match($line, '([0-9]+)')

    if ($match.Success) {
        $embedding = [int]$match.Groups[1].Value
    }
}
        if ($line -like "*quantization*") {
            $quantization = ($line -split '\s+')[-1]
        }
    }

    Log "Architecture: $architecture"
    Log "Parameters: $parameters B"
    Log "Quantization: $quantization"

    $ctx = TestContext $model

    if ($ctx -eq 0) {
        $ctx = 4096
    }

    $maxTokens = [int]($ctx / 4)

    Log "Context window: $ctx"
    Log "Max tokens: $maxTokens"

    $timing = BenchmarkGeneration $model

    Log "Load time: $($timing.loadTime)s"
    Log "First token latency: $($timing.firstToken)s"
    Log "Tokens/sec: $($timing.tokensPerSecond)"
    Log "GPU memory: $($timing.gpuMem) MB"
    Log "GPU util: $($timing.gpuUtil)%"
    Log "CPU util: $($timing.cpuUtil)%"

    $entry = @{
        id = $model
        name = $model
        family = $architecture
        parameters = $parameters
        architecture = $architecture
        reasoning = $false
        input = @("text")
        capabilities = @("completion")
        contextWindow = $ctx
        maxTokens = $maxTokens
        embeddingLength = $embedding
        quantization = $quantization
        api = "ollama"

        loadTimeSec = $timing.loadTime
        firstTokenLatencySec = $timing.firstToken
        tokensPerSecond = $timing.tokensPerSecond

        gpuMemoryMB = $timing.gpuMem
        gpuUtilizationPercent = $timing.gpuUtil
        cpuUtilizationPercent = $timing.cpuUtil
    }

    $results += $entry
}

Log "Writing benchmark report"

$json = $results | ConvertTo-Json -Depth 6
$json | Out-File $OutputFile -Encoding utf8

Log "Benchmark complete"
Log "Report written to $OutputFile"