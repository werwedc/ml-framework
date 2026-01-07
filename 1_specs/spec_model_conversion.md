# Spec: Model Conversion Utilities for Tensor Parallelism

## Overview
Implement utilities to automatically convert standard models to tensor-parallel versions. This includes analyzing model architecture to identify parallelizable layers, estimating memory requirements, and providing recommendations for TP configuration.

## Context
Users should be able to:
1. Take an existing standard model
2. Convert it to a TP version with minimal changes
3. Understand which layers are parallelized and how
4. Get estimates of memory and performance improvements

## Implementation Details

### 1. Layer Analysis Result

```csharp
namespace MLFramework.Conversion;

public class LayerAnalysisResult
{
    public string LayerName { get; set; } = "";
    public string LayerType { get; set; } = "";
    public bool IsParallelizable { get; set; }
    public ParallelismType SuggestedParallelism { get; set; }
    public long MemoryBytes { get; set; }
    public long ParameterCount { get; set; }

    public static LayerAnalysisResult NotParallelizable(
        string name,
        string type,
        long memoryBytes,
        long paramCount)
    {
        return new LayerAnalysisResult
        {
            LayerName = name,
            LayerType = type,
            IsParallelizable = false,
            SuggestedParallelism = ParallelismType.None,
            MemoryBytes = memoryBytes,
            ParameterCount = paramCount
        };
    }

    public static LayerAnalysisResult ColumnParallel(
        string name,
        string type,
        long memoryBytes,
        long paramCount)
    {
        return new LayerAnalysisResult
        {
            LayerName = name,
            LayerType = type,
            IsParallelizable = true,
            SuggestedParallelism = ParallelismType.Column,
            MemoryBytes = memoryBytes,
            ParameterCount = paramCount
        };
    }

    public static LayerAnalysisResult RowParallel(
        string name,
        string type,
        long memoryBytes,
        long paramCount)
    {
        return new LayerAnalysisResult
        {
            LayerName = name,
            LayerType = type,
            IsParallelizable = true,
            SuggestedParallelism = ParallelismType.Row,
            MemoryBytes = memoryBytes,
            ParameterCount = paramCount
        };
    }
}

public enum ParallelismType
{
    None,
    Column,
    Row,
    ConvOutput,
    ConvInput
}
```

### 2. Model Analysis Report

```csharp
public class TPAnalysisReport
{
    public List<LayerAnalysisResult> Layers { get; set; } = new();
    public long TotalMemoryBytes { get; set; }
    public long ParallelizableMemoryBytes { get; set; }
    public double ParallelizablePercentage => TotalMemoryBytes > 0
        ? (double)ParallelizableMemoryBytes / TotalMemoryBytes * 100
        : 0;

    public Dictionary<string, string> Recommendations { get; set; } = new();
    public int SuggestedWorldSize { get; set; }

    public void AddLayer(LayerAnalysisResult layer)
    {
        Layers.Add(layer);
        TotalMemoryBytes += layer.MemoryBytes;
        if (layer.IsParallelizable)
        {
            ParallelizableMemoryBytes += layer.MemoryBytes;
        }
    }

    public string GenerateSummary()
    {
        var sb = new StringBuilder();
        sb.AppendLine("=== TP Model Analysis ===");
        sb.AppendLine($"Total layers: {Layers.Count}");
        sb.AppendLine($"Parallelizable layers: {Layers.Count(l => l.IsParallelizable)}");
        sb.AppendLine($"Total memory: {TotalMemoryBytes / 1024 / 1024} MB");
        sb.AppendLine($"Parallelizable: {ParallelizableMemoryBytes / 1024 / 1024} MB ({ParallelizablePercentage:F1}%)");
        sb.AppendLine($"Suggested world size: {SuggestedWorldSize}");
        sb.AppendLine();

        sb.AppendLine("Recommendations:");
        foreach (var rec in Recommendations)
        {
            sb.AppendLine($"  - {rec.Key}: {rec.Value}");
        }

        return sb.ToString();
    }
}
```

### 3. Model Analyzer

```csharp
public static class TPModelAnalyzer
{
    /// <summary>
    /// Analyze a model to determine TP compatibility and configuration
    /// </summary>
    public static TPAnalysisReport Analyze(Module model, int maxWorldSize = 8)
    {
        var report = new TPAnalysisReport();
        AnalyzeModule(model, report, "");

        // Determine suggested world size
        report.SuggestedWorldSize = CalculateSuggestedWorldSize(report, maxWorldSize);

        // Generate recommendations
        GenerateRecommendations(report);

        return report;
    }

    private static void AnalyzeModule(Module module, TPAnalysisReport report, string prefix)
    {
        foreach (var param in module.Parameters)
        {
            AnalyzeParameter(param, report, prefix);
        }

        foreach (var submodule in module.Modules)
        {
            string newPrefix = string.IsNullOrEmpty(prefix)
                ? submodule.Name
                : $"{prefix}.{submodule.Name}";
            AnalyzeModule(submodule, report, newPrefix);
        }
    }

    private static void AnalyzeParameter(Parameter param, TPAnalysisReport report, string prefix)
    {
        if (param.Data == null)
            return;

        string fullName = string.IsNullOrEmpty(prefix) ? param.Name : $"{prefix}.{param.Name}";
        long memoryBytes = param.Data.MemoryBytes;
        long paramCount = param.Data.Shape.Aggregate(1L, (a, b) => a * b);

        // Determine if layer is parallelizable based on type and shape
        var result = DetermineParallelism(param, fullName, memoryBytes, paramCount);
        report.AddLayer(result);
    }

    private static LayerAnalysisResult DetermineParallelism(
        Parameter param,
        string fullName,
        long memoryBytes,
        long paramCount)
    {
        // Check if parameter name indicates layer type
        string lowerName = fullName.ToLowerInvariant();

        // Linear layers
        if (lowerName.Contains("weight") && param.Data.Shape.Length == 2)
        {
            int outFeatures = param.Data.Shape[0];
            int inFeatures = param.Data.Shape[1];

            // Large output dimension suggests column parallelism
            if (outFeatures > inFeatures * 2)
            {
                return LayerAnalysisResult.ColumnParallel(fullName, "Linear", memoryBytes, paramCount);
            }
            // Large input dimension suggests row parallelism
            else if (inFeatures > outFeatures * 2)
            {
                return LayerAnalysisResult.RowParallel(fullName, "Linear", memoryBytes, paramCount);
            }
        }

        // Conv2d layers
        if (lowerName.Contains("conv") && param.Data.Shape.Length == 4)
        {
            int outChannels = param.Data.Shape[0];
            int inChannels = param.Data.Shape[1];

            // Output channel parallelism is more common
            if (outChannels > 64) // Threshold for parallelism
            {
                return LayerAnalysisResult.ColumnParallel(fullName, "Conv2d", memoryBytes, paramCount);
            }
        }

        // Not parallelizable (e.g., embeddings, layer norm, etc.)
        return LayerAnalysisResult.NotParallelizable(fullName, "Unknown", memoryBytes, paramCount);
    }

    private static int CalculateSuggestedWorldSize(TPAnalysisReport report, int maxWorldSize)
    {
        if (report.ParallelizableMemoryBytes == 0)
            return 1;

        // Simple heuristic: divide memory by target per-device memory
        // Assume target of 4GB per device for now
        const long targetMemoryPerDevice = 4L * 1024 * 1024 * 1024;

        int suggestedWorldSize = (int)Math.Ceiling(
            (double)report.ParallelizableMemoryBytes / targetMemoryPerDevice);

        return Math.Clamp(suggestedWorldSize, 1, maxWorldSize);
    }

    private static void GenerateRecommendations(TPAnalysisReport report)
    {
        report.Recommendations.Clear();

        if (report.ParallelizablePercentage > 80)
        {
            report.Recommendations["Parallelization"] = "Highly recommended for this model";
        }
        else if (report.ParallelizablePercentage > 50)
        {
            report.Recommendations["Parallelization"] = "Recommended with significant memory savings";
        }
        else if (report.ParallelizablePercentage > 20)
        {
            report.Recommendations["Parallelization"] = "Moderate benefit, consider hybrid approach";
        }
        else
        {
            report.Recommendations["Parallelization"] = "Low benefit, consider other strategies";
        }

        if (report.SuggestedWorldSize > 4)
        {
            report.Recommendations["World Size"] =
                $"Large world size suggested ({report.SuggestedWorldSize}). Consider multi-node setup.";
        }
    }
}
```

### 4. Model Converter

```csharp
public static class TPModelConverter
{
    /// <summary>
    /// Convert a standard model to tensor-parallel version
    /// </summary>
    public static Module ConvertToTP(
        Module model,
        int worldSize,
        TensorParallelGroup? processGroup = null)
    {
        return ConvertModuleRecursive(model, worldSize, processGroup, "");
    }

    private static Module ConvertModuleRecursive(
        Module module,
        int worldSize,
        TensorParallelGroup? processGroup,
        string prefix)
    {
        // Check if this module is a Linear layer
        if (module is LinearLayer linearLayer)
        {
            return ConvertLinearLayer(linearLayer, worldSize, processGroup, prefix);
        }

        // Check if this module is a Conv2d layer
        if (module is Conv2dLayer convLayer)
        {
            return ConvertConv2dLayer(convLayer, worldSize, processGroup, prefix);
        }

        // For other modules, recursively convert submodules
        // Note: This would require Module to support replacing submodules
        // For now, return the module as-is
        return module;
    }

    private static Module ConvertLinearLayer(
        LinearLayer linearLayer,
        int worldSize,
        TensorParallelGroup? processGroup,
        string prefix)
    {
        string fullName = string.IsNullOrEmpty(prefix) ? linearLayer.Name : $"{prefix}.{linearLayer.Name}";

        // Determine parallelism strategy based on analysis
        var param = linearLayer.Parameters.FirstOrDefault(p => p.Name == "weight");
        if (param?.Data == null)
            return linearLayer;

        int outFeatures = param.Data.Shape[0];
        int inFeatures = param.Data.Shape[1];

        // Copy weights and bias from original layer
        Tensor weight = param.Data.Clone();
        Tensor? bias = linearLayer.Parameters.FirstOrDefault(p => p.Name == "bias")?.Data?.Clone();

        // Decide on parallelism strategy
        if (outFeatures > inFeatures * 2)
        {
            // Column parallel
            return CreateColumnParallelFromWeights(
                weight, bias, inFeatures, outFeatures, worldSize, processGroup);
        }
        else if (inFeatures > outFeatures * 2)
        {
            // Row parallel
            return CreateRowParallelFromWeights(
                weight, bias, inFeatures, outFeatures, worldSize, processGroup);
        }

        // Not clearly parallelizable, return original
        return linearLayer;
    }

    private static Module CreateColumnParallelFromWeights(
        Tensor fullWeight,
        Tensor? fullBias,
        int inFeatures,
        int outFeatures,
        int worldSize,
        TensorParallelGroup? processGroup)
    {
        // Create new column-parallel layer
        var layer = new ColumnParallelLinear(
            inFeatures, outFeatures, bias: fullBias != null,
            gatherOutput: false, processGroup: processGroup);

        // Shard the weight
        int shardOutFeatures = outFeatures / worldSize;
        int rank = TensorParallel.GetRank();
        int startIdx = rank * shardOutFeatures;
        int endIdx = startIdx + shardOutFeatures;

        // Slice weight to local shard
        var weightShard = fullWeight.Slice(0, startIdx, endIdx);

        // Copy sharded weight to layer
        // This would require accessing the layer's weight parameter
        // For now, assume there's a SetWeight method or similar
        // layer.SetWeight(weightShard);

        // If bias exists, shard it too
        if (fullBias != null)
        {
            var biasShard = fullBias.Slice(0, startIdx, endIdx);
            // layer.SetBias(biasShard);
        }

        return layer;
    }

    private static Module CreateRowParallelFromWeights(
        Tensor fullWeight,
        Tensor? fullBias,
        int inFeatures,
        int outFeatures,
        int worldSize,
        TensorParallelGroup? processGroup)
    {
        // Create new row-parallel layer
        var layer = new RowParallelLinear(
            inFeatures, outFeatures, bias: fullBias != null,
            inputIsSharded: true, processGroup: processGroup);

        // Shard the weight along row dimension
        int shardInFeatures = inFeatures / worldSize;
        int rank = TensorParallel.GetRank();
        int startIdx = rank * shardInFeatures;
        int endIdx = startIdx + shardInFeatures;

        // Slice weight to local shard
        var weightShard = fullWeight.Slice(1, startIdx, endIdx);

        // Copy sharded weight to layer
        // layer.SetWeight(weightShard);

        // Bias is not sharded for row parallel
        if (fullBias != null)
        {
            // layer.SetBias(fullBias);
        }

        return layer;
    }

    private static Module ConvertConv2dLayer(
        Conv2dLayer convLayer,
        int worldSize,
        TensorParallelGroup? processGroup,
        string prefix)
    {
        // Similar logic for Conv2d layers
        // For now, return original
        return convLayer;
    }
}
```

### 5. Memory Estimator

```csharp
public static class TPMemoryEstimator
{
    /// <summary>
    /// Estimate memory usage for model with and without TP
    /// </summary>
    public static MemoryEstimate EstimateMemory(Module model, int worldSize)
    {
        var analysis = TPModelAnalyzer.Analyze(model, worldSize);
        var estimate = new MemoryEstimate();

        // Base memory (without TP)
        estimate.BaseMemoryMB = analysis.TotalMemoryBytes / 1024 / 1024;

        // Memory with TP (each rank stores only a fraction)
        estimate.TPMemoryPerRankMB = (
            analysis.TotalMemoryBytes - analysis.ParallelizableMemoryBytes +
            analysis.ParallelizableMemoryBytes / worldSize
        ) / 1024 / 1024;

        // Communication overhead (temporary buffers)
        estimate.CommunicationOverheadMB = estimate.TPMemoryPerRankMB * 0.1; // 10% overhead

        // Total memory per rank
        estimate.TotalMemoryPerRankMB = estimate.TPMemoryPerRankMB + estimate.CommunicationOverheadMB;

        // Memory savings
        estimate.MemorySavingsPercentage = (
            1 - (double)estimate.TotalMemoryPerRankMB / estimate.BaseMemoryMB
        ) * 100;

        return estimate;
    }

    public class MemoryEstimate
    {
        public long BaseMemoryMB { get; set; }
        public long TPMemoryPerRankMB { get; set; }
        public long CommunicationOverheadMB { get; set; }
        public long TotalMemoryPerRankMB { get; set; }
        public double MemorySavingsPercentage { get; set; }

        public string GenerateSummary()
        {
            return $"Base memory: {BaseMemoryMB} MB\n" +
                   $"TP memory per rank: {TotalMemoryPerRankMB} MB\n" +
                   $"Savings: {MemorySavingsPercentage:F1}%\n" +
                   $"Communication overhead: {CommunicationOverheadMB} MB";
        }
    }
}
```

## Files to Create

### Source Files
- `src/MLFramework/Conversion/LayerAnalysisResult.cs`
- `src/MLFramework/Conversion/TPAnalysisReport.cs`
- `src/MLFramework/Conversion/TPModelAnalyzer.cs`
- `src/MLFramework/Conversion/TPModelConverter.cs`
- `src/MLFramework/Conversion/TPMemoryEstimator.cs`

### Test Files
- `tests/MLFramework.Tests/Conversion/TPModelAnalyzerTests.cs`
- `tests/MLFramework.Tests/Conversion/TPMemoryEstimatorTests.cs`

## Test Requirements

1. **Model Analyzer Tests**
   - Test analysis of simple MLP models
   - Test identification of parallelizable layers
   - Test memory calculations
   - Test world size suggestions

2. **Model Converter Tests**
   - Test conversion of Linear layers
   - Test weight sharding is correct
   - Test converted models produce same outputs as original
   - Test bias handling

3. **Memory Estimator Tests**
   - Test memory estimates for different models
   - Test memory savings calculations
   - Test communication overhead estimates

4. **Integration Tests**
   - Test full analysis → conversion pipeline
   - Test with different model architectures
   - Test edge cases (non-divisible dimensions)

## Dependencies
- `Module`, `LinearLayer`, `Conv2dLayer`, `Parameter` from framework
- `TensorParallel` context manager
- `ColumnParallelLinear`, `RowParallelLinear` from TP layers
- Tensor operations (Clone, Slice, etc.)

## Success Criteria
- [ ] Model analyzer correctly identifies parallelizable layers
- [ ] Analysis reports include accurate memory estimates
- [ ] World size suggestions are reasonable
- [ ] Model converter creates valid TP layers
- [ ] Converted models have correctly sharded weights
- [ ] Memory estimator provides useful estimates
- [ ] Unit tests pass for all scenarios
- [ ] Integration test demonstrates full conversion pipeline

## Estimated Time
45-60 minutes
