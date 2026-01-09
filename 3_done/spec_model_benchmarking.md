# Spec: Model Benchmarking Framework

## Overview
Implement a benchmarking system to evaluate model performance on datasets.

## Requirements

### 1. BenchmarkOptions Class
Benchmark configuration:
- Dataset (string): Dataset name or path
- Subset (string): "train", "val", "test", or custom subset
- BatchSize (int): Batch size for inference
- NumBatches (int): Number of batches to benchmark (null = all)
- NumIterations (int): Number of iterations per batch (for averaging)
- Device (Device): Device to run on (default: GPU if available)
- WarmupIterations (int): Warmup iterations before measurement (default: 10)
- Preprocess (bool): Whether to include preprocessing time
- Postprocess (bool): Whether to include postprocessing time
- IncludeMemoryProfile (bool): Whether to profile memory usage

### 2. BenchmarkResult Class
Benchmark results:
- ModelName (string): Model identifier
- Dataset (string): Dataset used
- TotalSamples (int): Number of samples processed
- Throughput (float): Samples per second
- AvgLatency (float): Average latency per sample (ms)
- MinLatency (float): Minimum latency (ms)
- MaxLatency (float): Maximum latency (ms)
- P50Latency (float): 50th percentile latency (ms)
- P95Latency (float): 95th percentile latency (ms)
- P99Latency (float): 99th percentile latency (ms)
- MemoryPeak (long): Peak memory usage (bytes)
- MemoryAvg (long): Average memory usage (bytes)
- Accuracy (float): Accuracy on dataset (if labels available)
- BenchmarkDuration (TimeSpan): Total benchmark time
- Timestamp (DateTime): When benchmark was run

### 3. ComparisonOptions Class
Comparison configuration:
- Models (string[]): List of model names to compare
- Dataset (string): Dataset to use
- Subset (string): Dataset subset
- BatchSize (int): Batch size
- Device (Device): Device to run on
- Metrics (string[]): Metrics to compare (latency, throughput, accuracy, memory)

### 4. ComparisonResult Class
Comparison results:
- ModelResults (Dictionary<string, BenchmarkResult>): Results for each model
- Winner (string): Best model based on primary metric
- RankByMetric (Dictionary<string, string[]>): Rankings by each metric
- StatisticalSignificance (Dictionary<string, bool>): Whether differences are significant

### 5. ModelBenchmarkService
Benchmark operations:
- `Benchmark(string modelName, BenchmarkOptions options)`: Run single model benchmark
- `BenchmarkModel(Model model, BenchmarkOptions options)`: Benchmark model object directly
- `Compare(string[] modelNames, ComparisonOptions options)`: Compare multiple models
- `BenchmarkBatch(Dictionary<string, BenchmarkOptions> benchmarks)`: Run multiple benchmarks in parallel

### 6. BenchmarkHistory Class
Store and retrieve historical benchmarks:
- `SaveResult(BenchmarkResult result)`: Save benchmark result
- `GetHistory(string modelName, int limit = 10)`: Get recent benchmarks for a model
- `GetLatest(string modelName)`: Get latest benchmark
- `CompareWithPrevious(BenchmarkResult result)`: Compare with previous run

### 7. BenchmarkReporter
Generate reports:
- `PrintResult(BenchmarkResult result)`: Console output
- `PrintComparison(ComparisonResult result)`: Console comparison table
- `GenerateReport(BenchmarkResult result, string outputPath)`: Save report to file
- `GenerateComparisonReport(ComparisonResult result, string outputPath)`: Save comparison to file
- Format options: Plain text, Markdown, JSON, CSV

### 8. Unit Tests
Test cases for:
- Single model benchmark
- Comparison of multiple models
- Warmup iterations don't affect results
- Latency percentiles are calculated correctly
- Memory profiling
- Accuracy calculation
- Benchmark history storage and retrieval
- Report generation (text, markdown, JSON)
- Edge cases (empty dataset, single sample, zero iterations)

## Files to Create
- `src/ModelZoo/Benchmark/BenchmarkOptions.cs`
- `src/ModelZoo/Benchmark/BenchmarkResult.cs`
- `src/ModelZoo/Benchmark/ComparisonOptions.cs`
- `src/ModelZoo/Benchmark/ComparisonResult.cs`
- `src/ModelZoo/Benchmark/ModelBenchmarkService.cs`
- `src/ModelZoo/Benchmark/BenchmarkHistory.cs`
- `src/ModelZoo/Benchmark/BenchmarkReporter.cs`
- `tests/ModelZooTests/Benchmark/BenchmarkTests.cs`

## Dependencies
- `ModelZoo` (from spec_model_zoo_load_api.md)
- `ModelRegistry` (from spec_model_registry.md)
- Existing Dataset/DataLoader infrastructure (if available)

## Success Criteria
- Can benchmark models accurately
- Latency measurements are consistent across runs
- Percentiles are calculated correctly
- Reports are clear and informative
- Test coverage > 85%
