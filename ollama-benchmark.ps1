param(
    [string]$ModelId,
    [string]$OutputPath = "model-benchmark.json"
)

$ErrorActionPreference = "Stop"

$OllamaUrl = "http://localhost:11434/api/generate"
$OutputFile = $OutputPath

$ContextTests = @(8192, 16384, 32768, 65536, 131072)
$DefaultContextWindow = 4096
$BenchmarkPrompt = "Explain the CAP theorem in distributed systems."
$BenchmarkNumPredict = 200

function Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] $Message"
}

function Invoke-OllamaGenerate {
    param(
        [Parameter(Mandatory = $true)][string]$Model,
        [Parameter(Mandatory = $true)][string]$Prompt,
        [hashtable]$Options = @{}
    )

    $body = @{
        model   = $Model
        prompt  = $Prompt
        stream  = $false
        options = $Options
    } | ConvertTo-Json -Depth 10

    return Invoke-RestMethod -Uri $OllamaUrl -Method Post -Body $body -ContentType "application/json"
}

function Get-GpuSnapshot {
    try {
        $raw = nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
        if (-not $raw) {
            throw "Empty nvidia-smi output"
        }

        $firstLine = ($raw -split "`r?`n" | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 1)
        $parts = $firstLine.Split(",")

        return @{
            util = [int]$parts[0].Trim()
            mem  = [int]$parts[1].Trim()
        }
    }
    catch {
        return @{ util = 0; mem = 0 }
    }
}

function Get-CpuSnapshot {
    try {
        $cpu = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
        return [math]::Round($cpu, 1)
    }
    catch {
        return 0
    }
}

function Parse-ModelfileOptions {
    param([string]$Modelfile)

    if ([string]::IsNullOrWhiteSpace($Modelfile)) {
        return @()
    }

    $names = @()
    $lines = $Modelfile -split "`r?`n"

    foreach ($line in $lines) {
        if ($line -match '^\s*PARAMETER\s+([^\s]+)\s+') {
            $names += $matches[1]
        }
    }

    return $names | Sort-Object -Unique
}

function Parse-ModelParameters {
    param([string]$ParametersText)

    if ([string]::IsNullOrWhiteSpace($ParametersText)) {
        return @{}
    }

    $map = @{}
    $lines = $ParametersText -split "`r?`n"

    foreach ($line in $lines) {
        if ($line -match '^\s*([^\s]+)\s+(.+?)\s*$') {
            $key = $matches[1]
            $value = $matches[2]
            if (-not $map.ContainsKey($key)) {
                $map[$key] = $value
            }
        }
    }

    return $map
}

function Get-InstalledModels {
    $jsonOutput = $null

    try {
        $jsonOutput = ollama list --json 2>$null
    }
    catch {
        $jsonOutput = $null
    }

    if ($jsonOutput) {
        $parsed = $jsonOutput | ConvertFrom-Json

        if ($parsed -is [array]) {
            return $parsed
        }

        return @($parsed)
    }

    Log "'ollama list --json' unsupported, falling back to plain 'ollama list' output parsing"

    $listOutput = ollama list
    if (-not $listOutput) {
        throw "Unable to read installed models from 'ollama list'."
    }

    $models = @()
    $lines = $listOutput -split "`r?`n"

    foreach ($line in $lines) {
        $trimmed = $line.Trim()
        if (-not $trimmed) { continue }
        if ($trimmed -match '^NAME\s+') { continue }

        if ($trimmed -match '^([^\s]+)\s+') {
            $modelName = $matches[1]
            $models += [PSCustomObject]@{ name = $modelName }
        }
    }

    if ($models.Count -eq 0) {
        throw "Unable to parse installed models from 'ollama list' output."
    }

    return $models
}

function Get-ModelShowJson {
    param([Parameter(Mandatory = $true)][string]$Model)

    $showJson = ollama show $Model --json 2>$null
    if (-not $showJson) {
        throw "Unable to fetch metadata for model '$Model' via 'ollama show --json'."
    }

    return $showJson | ConvertFrom-Json
}

function Get-ModelInfoValue {
    param(
        [Parameter(Mandatory = $true)]$ModelInfo,
        [Parameter(Mandatory = $true)][string[]]$PreferredKeys,
        [string]$Suffix
    )

    foreach ($key in $PreferredKeys) {
        if ($ModelInfo.PSObject.Properties.Name -contains $key) {
            return $ModelInfo.$key
        }
    }

    if ($Suffix) {
        $candidate = $ModelInfo.PSObject.Properties.Name | Where-Object { $_ -like "*$Suffix" } | Select-Object -First 1
        if ($candidate) {
            return $ModelInfo.$candidate
        }
    }

    return $null
}

function Get-ModelMetadata {
    param([Parameter(Mandatory = $true)][string]$Model)

    $show = Get-ModelShowJson -Model $Model

    $modelInfo = $show.model_info
    $details = $show.details

    $architecture = ""
    $parameters = 0
    $embeddingLength = 0
    $contextLength = 0
    $quantization = ""

    if ($modelInfo) {
        $archValue = Get-ModelInfoValue -ModelInfo $modelInfo -PreferredKeys @('general.architecture')
        if ($archValue) { $architecture = [string]$archValue }

        $paramCount = Get-ModelInfoValue -ModelInfo $modelInfo -PreferredKeys @('general.parameter_count')
        if ($paramCount) { $parameters = [math]::Round(([double]$paramCount) / 1e9, 3) }

        $embedValue = Get-ModelInfoValue -ModelInfo $modelInfo -PreferredKeys @('llama.embedding_length', 'qwen2.embedding_length', 'gemma.embedding_length') -Suffix '.embedding_length'
        if ($embedValue) { $embeddingLength = [int]$embedValue }

        $ctxValue = Get-ModelInfoValue -ModelInfo $modelInfo -PreferredKeys @('llama.context_length', 'qwen2.context_length', 'gemma.context_length') -Suffix '.context_length'
        if ($ctxValue) { $contextLength = [int]$ctxValue }

        $fileType = Get-ModelInfoValue -ModelInfo $modelInfo -PreferredKeys @('general.file_type')
        if ($fileType) { $quantization = [string]$fileType }
    }

    if ($details) {
        if (-not $architecture -and $details.family) { $architecture = [string]$details.family }
        if ($parameters -eq 0 -and $details.parameter_size -and $details.parameter_size -match '([0-9]+\.?[0-9]*)') {
            $parameters = [double]$matches[1]
        }
        if (-not $quantization -and $details.quantization_level) {
            $quantization = [string]$details.quantization_level
        }
    }

    $capabilities = @()
    if ($show.capabilities) {
        $capabilities = @($show.capabilities)
    }
    elseif ($show.details -and $show.details.families) {
        $capabilities = @("completion")
    }

    if ($capabilities.Count -eq 0) {
        $capabilities = @("completion")
    }

    $inputTypes = @("text")
    if ($capabilities -contains "vision") {
        $inputTypes += "image"
    }

    $outputTypes = @("text")
    if ($capabilities -contains "embedding") {
        $outputTypes += "vector"
    }

    $acceptsTools = $capabilities -contains "tools"

    $reasoning = $false
    if ($model -match '(reason|r1|thinking)') {
        $reasoning = $true
    }

    $optionsMap = Parse-ModelParameters -ParametersText ([string]$show.parameters)
    $modelfileOptions = Parse-ModelfileOptions -Modelfile ([string]$show.modelfile)

    return @{
        architecture = $architecture
        parameters = $parameters
        embeddingLength = $embeddingLength
        contextLength = $contextLength
        quantization = $quantization
        capabilities = $capabilities
        inputTypes = ($inputTypes | Sort-Object -Unique)
        outputTypes = ($outputTypes | Sort-Object -Unique)
        acceptsTools = $acceptsTools
        reasoning = $reasoning
        options = $optionsMap
        optionsAvailable = $modelfileOptions
    }
}

function Test-ContextWindow {
    param([Parameter(Mandatory = $true)][string]$Model)

    $maxStable = 0

    foreach ($ctx in $ContextTests) {
        Log "Testing context $ctx"
        try {
            $response = Invoke-OllamaGenerate -Model $Model -Prompt "hello" -Options @{ num_ctx = $ctx; num_predict = 8 }
            if ($response.response) {
                $maxStable = $ctx
                Log "success"
            }
        }
        catch {
            Log "failed"
            break
        }
    }

    if ($maxStable -eq 0) {
        return $DefaultContextWindow
    }

    return $maxStable
}

function Get-EfficiencyProfile {
    param(
        [double]$TokensPerSecond,
        [double]$FirstTokenLatencySec,
        [double]$LoadTimeSec,
        [double]$ParametersB,
        [int]$GpuMemoryMB
    )

    $perB = 0
    if ($ParametersB -gt 0) {
        $perB = [math]::Round($TokensPerSecond / $ParametersB, 3)
    }

    $perGBVram = 0
    if ($GpuMemoryMB -gt 0) {
        $perGBVram = [math]::Round($TokensPerSecond / ($GpuMemoryMB / 1024.0), 3)
    }

    $responseEfficiencyScore = 0
    if ($TokensPerSecond -gt 0) {
        $responseEfficiencyScore = [math]::Round($TokensPerSecond / (1 + $FirstTokenLatencySec + ($LoadTimeSec / 2)), 3)
    }

    return @{
        tokensPerSecondPerBParameter = $perB
        tokensPerSecondPerGBVram = $perGBVram
        responseEfficiencyScore = $responseEfficiencyScore
    }
}

function Get-WorkloadRecommendations {
    param(
        [double]$TokensPerSecond,
        [double]$FirstTokenLatencySec,
        [int]$ContextWindow,
        [bool]$AcceptsTools,
        [string[]]$Capabilities,
        [bool]$Reasoning
    )

    $recommendations = @()

    if ($FirstTokenLatencySec -le 1.0 -and $TokensPerSecond -ge 25) {
        $recommendations += "interactive chat and coding assistants"
    }

    if ($ContextWindow -ge 65536) {
        $recommendations += "long-context summarization and retrieval workflows"
    }

    if ($Reasoning) {
        $recommendations += "multi-step reasoning tasks"
    }

    if ($AcceptsTools) {
        $recommendations += "tool-augmented agents and function-calling workflows"
    }

    if ($Capabilities -contains "vision") {
        $recommendations += "multimodal image+text analysis"
    }

    if ($recommendations.Count -eq 0) {
        $recommendations += "general-purpose text generation"
    }

    return $recommendations | Sort-Object -Unique
}

function Get-OptimalRuntimeSettings {
    param(
        [int]$ContextWindow,
        [double]$TokensPerSecond,
        [double]$FirstTokenLatencySec,
        [hashtable]$ModelOptions
    )

    $maxTokens = [int]($ContextWindow / 4)

    $qualityTemperature = 0.7
    if ($ModelOptions.ContainsKey("temperature")) {
        try {
            $qualityTemperature = [double]$ModelOptions["temperature"]
        }
        catch {
            $qualityTemperature = 0.7
        }
    }

    $latencyProfile = "balanced"
    if ($FirstTokenLatencySec -gt 1.5 -or $TokensPerSecond -lt 15) {
        $latencyProfile = "throughput-conservative"
    }
    elseif ($FirstTokenLatencySec -lt 0.5 -and $TokensPerSecond -gt 35) {
        $latencyProfile = "low-latency"
    }

    $recommendedNumPredict = [math]::Min($maxTokens, 1024)

    return @{
        profile = $latencyProfile
        recommendedOptions = @{
            num_ctx = $ContextWindow
            num_predict = $recommendedNumPredict
            temperature = $qualityTemperature
        }
    }
}

function Benchmark-Generation {
    param([Parameter(Mandatory = $true)][string]$Model)

    Log "Running performance benchmark"

    $gpuBefore = Get-GpuSnapshot
    $cpuBefore = Get-CpuSnapshot

    $response = Invoke-OllamaGenerate -Model $Model -Prompt $BenchmarkPrompt -Options @{ num_predict = $BenchmarkNumPredict }

    $gpuAfter = Get-GpuSnapshot
    $cpuAfter = Get-CpuSnapshot

    $loadSec = [double]($response.load_duration) / 1e9
    $promptEvalSec = [double]($response.prompt_eval_duration) / 1e9
    $evalSec = [double]($response.eval_duration) / 1e9
    $evalCount = [int]$response.eval_count

    $tokensPerSecond = 0
    if ($evalSec -gt 0 -and $evalCount -gt 0) {
        $tokensPerSecond = [math]::Round($evalCount / $evalSec, 2)
    }

    $firstTokenLatencySec = [math]::Round($loadSec + $promptEvalSec, 3)

    return @{
        loadTime = [math]::Round($loadSec, 3)
        firstToken = $firstTokenLatencySec
        tokensPerSecond = $tokensPerSecond
        generatedTokens = $evalCount
        promptTokens = [int]$response.prompt_eval_count
        evalDurationSec = [math]::Round($evalSec, 3)
        gpuMem = [math]::Max($gpuBefore.mem, $gpuAfter.mem)
        gpuUtil = [math]::Max($gpuBefore.util, $gpuAfter.util)
        cpuUtil = [math]::Round(([math]::Max($cpuBefore, $cpuAfter)), 1)
    }
}

Log "Discovering installed models"
$allModels = Get-InstalledModels

$selectedModels = @()
if ($ModelId) {
    $match = $allModels | Where-Object { $_.name -eq $ModelId }
    if (-not $match) {
        throw "Requested model '$ModelId' was not found in installed models."
    }

    $selectedModels = @($match)
    Log "Single-model mode enabled for: $ModelId"
}
else {
    $selectedModels = $allModels
}

Log "$($selectedModels.Count) models queued for benchmarking"

$results = @()

foreach ($modelEntry in $selectedModels) {
    $model = [string]$modelEntry.name

    if ([string]::IsNullOrWhiteSpace($model)) {
        continue
    }

    Log ""
    Log "Benchmarking $model"

    $meta = Get-ModelMetadata -Model $model

    Log "Architecture: $($meta.architecture)"
    Log "Parameters: $($meta.parameters) B"
    Log "Quantization: $($meta.quantization)"
    Log "Capabilities: $([string]::Join(', ', $meta.capabilities))"

    $contextWindow = Test-ContextWindow -Model $model

    if ($meta.contextLength -gt 0) {
        $contextWindow = [math]::Min($contextWindow, $meta.contextLength)
    }

    $maxTokens = [int]($contextWindow / 4)

    Log "Context window: $contextWindow"
    Log "Max tokens: $maxTokens"

    $timing = Benchmark-Generation -Model $model

    $efficiency = Get-EfficiencyProfile `
        -TokensPerSecond $timing.tokensPerSecond `
        -FirstTokenLatencySec $timing.firstToken `
        -LoadTimeSec $timing.loadTime `
        -ParametersB $meta.parameters `
        -GpuMemoryMB $timing.gpuMem

    $optimalRuntime = Get-OptimalRuntimeSettings `
        -ContextWindow $contextWindow `
        -TokensPerSecond $timing.tokensPerSecond `
        -FirstTokenLatencySec $timing.firstToken `
        -ModelOptions $meta.options

    $workloadRecommendations = Get-WorkloadRecommendations `
        -TokensPerSecond $timing.tokensPerSecond `
        -FirstTokenLatencySec $timing.firstToken `
        -ContextWindow $contextWindow `
        -AcceptsTools $meta.acceptsTools `
        -Capabilities $meta.capabilities `
        -Reasoning $meta.reasoning

    Log "Load time: $($timing.loadTime)s"
    Log "First token latency: $($timing.firstToken)s"
    Log "Generated tokens: $($timing.generatedTokens)"
    Log "Tokens/sec: $($timing.tokensPerSecond)"
    Log "Efficiency score: $($efficiency.responseEfficiencyScore)"
    Log "GPU memory: $($timing.gpuMem) MB"
    Log "GPU util: $($timing.gpuUtil)%"
    Log "CPU util: $($timing.cpuUtil)%"

    $results += @{
        id = $model
        name = $model
        family = $meta.architecture
        parameters = $meta.parameters
        architecture = $meta.architecture
        reasoning = $meta.reasoning
        input = $meta.inputTypes
        output = $meta.outputTypes
        capabilities = $meta.capabilities
        acceptsTools = $meta.acceptsTools
        contextWindow = $contextWindow
        maxTokens = $maxTokens
        embeddingLength = $meta.embeddingLength
        quantization = $meta.quantization
        api = "ollama"

        optionsAvailable = $meta.optionsAvailable
        defaultOptions = $meta.options
        optimalRuntimeSettings = $optimalRuntime
        recommendedUseCases = $workloadRecommendations

        loadTimeSec = $timing.loadTime
        firstTokenLatencySec = $timing.firstToken
        tokensPerSecond = $timing.tokensPerSecond
        generatedTokens = $timing.generatedTokens
        promptTokens = $timing.promptTokens
        evalDurationSec = $timing.evalDurationSec

        efficiency = $efficiency

        gpuMemoryMB = $timing.gpuMem
        gpuUtilizationPercent = $timing.gpuUtil
        cpuUtilizationPercent = $timing.cpuUtil
    }
}

Log "Writing benchmark report"
$results | ConvertTo-Json -Depth 12 | Out-File $OutputFile -Encoding utf8

Log "Benchmark complete"
Log "Report written to $OutputFile"
