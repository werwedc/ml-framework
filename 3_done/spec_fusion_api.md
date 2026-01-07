# Spec: Fusion API and Attributes

## Overview
Implement the user-facing API for controlling fusion behavior, including attributes for explicit fusion hints and configuration options for controlling fusion granularity.

## Requirements

### 1. Fusible Attribute
Attribute to mark methods or operations as explicitly fusible.

```csharp
[AttributeUsage(AttributeTargets.Method | AttributeTargets.Class, AllowMultiple = false)]
public class FusibleAttribute : Attribute
{
    /// <summary>
    /// Creates a fusible attribute with default options
    /// </summary>
    public FusibleAttribute()
    {
        MaxOperations = 10;
        Strategy = FusionStrategy.Merge;
        Priority = 0;
    }

    /// <summary>
    /// Maximum number of operations to fuse
    /// </summary>
    public int MaxOperations { get; set; }

    /// <summary>
    /// Fusion strategy to apply
    /// </summary>
    public FusionStrategy Strategy { get; set; }

    /// <summary>
    /// Fusion priority (higher values = higher priority)
    /// </summary>
    public int Priority { get; set; }

    /// <summary>
    /// Specific fusion pattern to use (optional)
    /// </summary>
    public string? Pattern { get; set; }
}

// Examples of usage:

[Fusible]
public Tensor ProcessLayer(Tensor x)
{
    var y = x.Linear(128);
    y = y.ReLU();
    y = y.Dropout(0.1f);
    return y;
}

[Fusible(MaxOperations = 5, Priority = 10)]
public class HighPriorityFusionLayer
{
    public Tensor Forward(Tensor x)
    {
        var y = x.Conv2D(64, 3);
        y = y.BatchNorm();
        y = y.ReLU();
        return y;
    }
}
```

### 2. NoFusion Attribute
Attribute to prevent fusion of specific operations or methods.

```csharp
[AttributeUsage(AttributeTargets.Method | AttributeTargets.Class, AllowMultiple = false)]
public class NoFusionAttribute : Attribute
{
    /// <summary>
    /// Reason for preventing fusion (for logging/debugging)
    /// </summary>
    public string Reason { get; set; } = "Explicitly marked as non-fusible";
}

// Example usage:

[NoFusion]
public Tensor CustomOperation(Tensor x)
{
    // This operation will never be fused
    return x.ApplyCustomKernel();
}

[NoFusion(Reason = "Contains data-dependent control flow")]
public Tensor ConditionalOperation(Tensor x)
{
    if (x.Sum() > 0)
        return x.ReLU();
    else
        return x.Sigmoid();
}
```

### 3. Fusion Options
Configuration options for controlling fusion behavior globally.

```csharp
public static class GraphOptions
{
    /// <summary>
    /// Enables or disables fusion globally
    /// </summary>
    public static bool EnableFusion { get; set; } = true;

    /// <summary>
    /// Maximum number of operations to fuse together
    /// </summary>
    public static int MaxFusionOps { get; set; } = 10;

    /// <summary>
    /// Fusion backend to use
    /// </summary>
    public static FusionBackend FusionBackend { get; set; } = FusionBackend.Triton;

    /// <summary>
    /// Minimum benefit score required to apply fusion
    /// </summary>
    public static int MinBenefitScore { get; set; } = 50;

    /// <summary>
    /// Fusion aggressiveness level
    /// </summary>
    public static FusionAggressiveness Aggressiveness { get; set; } = FusionAggressiveness.Medium;

    /// <summary>
    /// Enable automatic fusion (default behavior)
    /// </summary>
    public static bool EnableAutomaticFusion { get; set; } = true;

    /// <summary>
    /// Enable fusion with user hints
    /// </summary>
    public static bool EnableHintedFusion { get; set; } = true;

    /// <summary>
    /// Enable autotuning for fused kernels
    /// </summary>
    public static bool EnableAutotuning { get; set; } = true;

    /// <summary>
    /// Enable BatchNorm folding into convolutions
    /// </summary>
    public static bool EnableBatchNormFolding { get; set; } = true;

    /// <summary>
    /// Enable Conv+Activation fusion
    /// </summary>
    public static bool EnableConvActivationFusion { get; set; } = true;

    /// <summary>
    /// Enable element-wise operation fusion
    /// </summary>
    public static bool EnableElementWiseFusion { get; set; } = true;

    /// <summary>
    /// Cache directory for autotuning results
    /// </summary>
    public static string? TuningCacheDirectory { get; set; }

    /// <summary>
    /// Resets all options to default values
    /// </summary>
    public static void ResetToDefaults()
    {
        EnableFusion = true;
        MaxFusionOps = 10;
        FusionBackend = FusionBackend.Triton;
        MinBenefitScore = 50;
        Aggressiveness = FusionAggressiveness.Medium;
        EnableAutomaticFusion = true;
        EnableHintedFusion = true;
        EnableAutotuning = true;
        EnableBatchNormFolding = true;
        EnableConvActivationFusion = true;
        EnableElementWiseFusion = true;
        TuningCacheDirectory = null;
    }
}

public enum FusionBackend
{
    Triton,
    XLA,
    Custom,
    None
}

public enum FusionAggressiveness
{
    Conservative,
    Medium,
    Aggressive
}
```

### 4. Fusion Context Manager
Context manager for temporarily modifying fusion options.

```csharp
public sealed class FusionContext : IDisposable
{
    private readonly FusionOptions _previousOptions;
    private readonly bool _restoreOnDispose;

    public FusionContext(FusionOptions options, bool restoreOnDispose = true)
    {
        _previousOptions = GetCurrentOptions();
        _restoreOnDispose = restoreOnDispose;
        SetCurrentOptions(options);
    }

    public void Dispose()
    {
        if (_restoreOnDispose)
        {
            SetCurrentOptions(_previousOptions);
        }
    }

    private static FusionOptions GetCurrentOptions()
    {
        return new FusionOptions
        {
            EnableFusion = GraphOptions.EnableFusion,
            MaxFusionOps = GraphOptions.MaxFusionOps,
            FusionBackend = GraphOptions.FusionBackend,
            MinBenefitScore = GraphOptions.MinBenefitScore,
            Aggressiveness = GraphOptions.Aggressiveness
        };
    }

    private static void SetCurrentOptions(FusionOptions options)
    {
        GraphOptions.EnableFusion = options.EnableFusion;
        GraphOptions.MaxFusionOps = options.MaxFusionOps;
        GraphOptions.FusionBackend = options.FusionBackend;
        GraphOptions.MinBenefitScore = options.MinBenefitScore;
        GraphOptions.Aggressiveness = options.Aggressiveness;
    }

    /// <summary>
    /// Creates a context with fusion disabled
    /// </summary>
    public static FusionContext DisableFusion()
    {
        return new FusionContext(new FusionOptions { EnableFusion = false });
    }

    /// <summary>
    /// Creates a context with custom fusion options
    /// </summary>
    public static FusionContext WithOptions(Action<FusionOptions> configure)
    {
        var options = GetCurrentOptions();
        configure(options);
        return new FusionContext(options);
    }
}

public record FusionOptions
{
    public bool EnableFusion { get; init; } = true;
    public int MaxFusionOps { get; init; } = 10;
    public FusionBackend FusionBackend { get; init; } = FusionBackend.Triton;
    public int MinBenefitScore { get; init; } = 50;
    public FusionAggressiveness Aggressiveness { get; init; } = FusionAggressiveness.Medium;
}

// Example usage:

public Tensor ModelForward(Tensor x)
{
    // Use aggressive fusion for this section
    using (FusionContext.WithOptions(opts =>
    {
        opts.Aggressiveness = FusionAggressiveness.Aggressive;
        opts.MaxFusionOps = 15;
    }))
    {
        var y = x.Conv2D(64, 3);
        y = y.BatchNorm();
        y = y.ReLU();
        y = y.Conv2D(128, 3);
        y = y.BatchNorm();
        y = y.ReLU();
        return y;
    }
}

public Tensor CustomForward(Tensor x)
{
    // Disable fusion for debugging
    using (FusionContext.DisableFusion())
    {
        var y = x.Linear(256);
        y = y.ReLU();
        return y;
    }
}
```

### 5. Fusion Statistics
API for querying fusion statistics and debugging.

```csharp
public interface IFusionStatistics
{
    /// <summary>
    /// Gets statistics for the current session
    /// </summary>
    FusionStatistics GetCurrentStatistics();

    /// <summary>
    /// Resets statistics
    /// </summary>
    void Reset();

    /// <summary>
    /// Logs fusion decisions for debugging
    /// </summary>
    void LogFusionDecisions();
}

public record FusionStatistics
{
    public required int TotalOperations { get; init; }
    public required int FusedOperations { get; init; }
    public required int FusedGroups { get; init; }
    public required double FusionPercentage { get; init; }
    public required int RejectedFusions { get; init; }
    public required double AverageOperationsPerFusedGroup { get; init; }
    public required IReadOnlyDictionary<FusionPatternType, int> PatternCounts { get; init; }
    public required IReadOnlyDictionary<string, int> RejectionReasons { get; init; }
    public required DateTime StartTime { get; init; }
    public required DateTime EndTime { get; init; }
}

public class FusionStatisticsCollector : IFusionStatistics
{
    private readonly object _lock = new();
    private int _totalOperations;
    private int _fusedOperations;
    private int _fusedGroups;
    private int _rejectedFusions;
    private readonly Dictionary<FusionPatternType, int> _patternCounts = new();
    private readonly Dictionary<string, int> _rejectionReasons = new();
    private DateTime _startTime = DateTime.UtcNow;
    private DateTime? _endTime;

    public FusionStatistics GetCurrentStatistics()
    {
        lock (_lock)
        {
            return new FusionStatistics
            {
                TotalOperations = _totalOperations,
                FusedOperations = _fusedOperations,
                FusedGroups = _fusedGroups,
                FusionPercentage = _totalOperations > 0
                    ? (_fusedOperations * 100.0 / _totalOperations)
                    : 0.0,
                RejectedFusions = _rejectedFusions,
                AverageOperationsPerFusedGroup = _fusedGroups > 0
                    ? (_fusedOperations * 1.0 / _fusedGroups)
                    : 0.0,
                PatternCounts = new Dictionary<FusionPatternType, int>(_patternCounts),
                RejectionReasons = new Dictionary<string, int>(_rejectionReasons),
                StartTime = _startTime,
                EndTime = _endTime ?? DateTime.UtcNow
            };
        }
    }

    public void Reset()
    {
        lock (_lock)
        {
            _totalOperations = 0;
            _fusedOperations = 0;
            _fusedGroups = 0;
            _rejectedFusions = 0;
            _patternCounts.Clear();
            _rejectionReasons.Clear();
            _startTime = DateTime.UtcNow;
            _endTime = null;
        }
    }

    public void LogFusionDecisions()
    {
        var stats = GetCurrentStatistics();

        Console.WriteLine("=== Fusion Statistics ===");
        Console.WriteLine($"Total Operations: {stats.TotalOperations}");
        Console.WriteLine($"Fused Operations: {stats.FusedOperations} ({stats.FusionPercentage:F2}%)");
        Console.WriteLine($"Fused Groups: {stats.FusedGroups}");
        Console.WriteLine($"Average Operations per Group: {stats.AverageOperationsPerFusedGroup:F2}");
        Console.WriteLine($"Rejected Fusions: {stats.RejectedFusions}");
        Console.WriteLine();

        Console.WriteLine("Pattern Distribution:");
        foreach (var (pattern, count) in stats.PatternCounts.OrderByDescending(kv => kv.Value))
        {
            Console.WriteLine($"  {pattern}: {count}");
        }

        Console.WriteLine();

        if (stats.RejectionReasons.Count > 0)
        {
            Console.WriteLine("Rejection Reasons:");
            foreach (var (reason, count) in stats.RejectionReasons.OrderByDescending(kv => kv.Value))
            {
                Console.WriteLine($"  {reason}: {count}");
            }
        }

        Console.WriteLine($"Duration: {(stats.EndTime - stats.StartTime).TotalSeconds:F2}s");
    }

    public void RecordOperation(Operation op)
    {
        lock (_lock)
        {
            _totalOperations++;
        }
    }

    public void RecordFusedGroup(IReadOnlyList<Operation> operations, FusionPatternType patternType)
    {
        lock (_lock)
        {
            _fusedOperations += operations.Count;
            _fusedGroups++;
            _patternCounts.TryGetValue(patternType, out var count);
            _patternCounts[patternType] = count + 1;
        }
    }

    public void RecordRejection(string reason)
    {
        lock (_lock)
        {
            _rejectedFusions++;
            _rejectionReasons.TryGetValue(reason, out var count);
            _rejectionReasons[reason] = count + 1;
        }
    }
}
```

### 6. Fusion API Helper Methods
Extension methods and helper functions for fusion API.

```csharp
public static class FusionApiExtensions
{
    /// <summary>
    /// Applies fusion to a computational graph
    /// </summary>
    public static ComputationalGraph ApplyFusion(
        this ComputationalGraph graph,
        IFusionEngine engine,
        FusionOptions? options = null)
    {
        var fusionOptions = options ?? new FusionOptions
        {
            EnableFusion = GraphOptions.EnableFusion,
            MaxFusionOps = GraphOptions.MaxFusionOps,
            MinBenefitScore = GraphOptions.MinBenefitScore,
            Aggressiveness = GraphOptions.Aggressiveness
        };

        var result = engine.ApplyFusion(graph, fusionOptions);
        return result.FusedGraph;
    }

    /// <summary>
    /// Gets all fused operations in the graph
    /// </summary>
    public static IReadOnlyList<FusedOperation> GetFusedOperations(this ComputationalGraph graph)
    {
        return graph.Operations.OfType<FusedOperation>().ToList();
    }

    /// <summary>
    /// Gets fusion statistics for a graph
    /// </summary>
    public static FusionGraphStatistics GetFusionStatistics(this ComputationalGraph graph)
    {
        var fusedOps = graph.GetFusedOperations();
        var totalOps = graph.Operations.Count;
        var fusedOpsCount = fusedOps.Sum(op => op.ConstituentOperations.Count);
        var rejectedOps = totalOps - fusedOps.Count;

        return new FusionGraphStatistics
        {
            TotalOperations = totalOps,
            FusedOperations = fusedOpsCount,
            FusedGroups = fusedOps.Count,
            RejectedOperations = rejectedOps,
            EstimatedSpeedup = EstimateSpeedup(fusedOps)
        };
    }

    private static double EstimateSpeedup(IReadOnlyList<FusedOperation> fusedOps)
    {
        // Simple heuristic: 1.3x speedup per fused operation group
        return fusedOps.Count * 1.3;
    }
}

public record FusionGraphStatistics
{
    public required int TotalOperations { get; init; }
    public required int FusedOperations { get; init; }
    public required int FusedGroups { get; init; }
    public required int RejectedOperations { get; init; }
    public required double EstimatedSpeedup { get; init; }
}
```

## Implementation Tasks

1. **Implement FusibleAttribute** (15 min)
   - Attribute definition
   - Properties for configuration
   - Documentation and examples

2. **Implement NoFusionAttribute** (10 min)
   - Attribute definition
   - Reason property
   - Documentation and examples

3. **Implement GraphOptions** (25 min)
   - Static options class
   - All fusion configuration options
   - ResetToDefaults method

4. **Implement FusionContext** (25 min)
   - Context manager class
   - IDisposable implementation
   - DisableFusion and WithOptions factory methods
   - Options state management

5. **Implement IFusionStatistics** (30 min)
   - FusionStatistics record
   - FusionStatisticsCollector class
   - Thread-safe statistics tracking
   - LogFusionDecisions method

6. **Implement FusionApiExtensions** (20 min)
   - ApplyFusion extension method
   - GetFusedOperations extension method
   - GetFusionStatistics extension method
   - Helper methods

## Test Cases

```csharp
[Test]
public void FusibleAttribute_ReadsProperties()
{
    var attr = new FusibleAttribute
    {
        MaxOperations = 5,
        Strategy = FusionStrategy.Fold,
        Priority = 10
    };

    Assert.AreEqual(5, attr.MaxOperations);
    Assert.AreEqual(FusionStrategy.Fold, attr.Strategy);
    Assert.AreEqual(10, attr.Priority);
}

[Test]
public void GraphOptions_ResetsToDefaults()
{
    GraphOptions.MaxFusionOps = 20;
    GraphOptions.EnableFusion = false;

    GraphOptions.ResetToDefaults();

    Assert.AreEqual(10, GraphOptions.MaxFusionOps);
    Assert.IsTrue(GraphOptions.EnableFusion);
}

[Test]
public void FusionContext_RestoresOptionsOnDispose()
{
    GraphOptions.MaxFusionOps = 10;

    using (FusionContext.WithOptions(opts => opts.MaxFusionOps = 20))
    {
        Assert.AreEqual(20, GraphOptions.MaxFusionOps);
    }

    Assert.AreEqual(10, GraphOptions.MaxFusionOps);
}

[Test]
public void FusionStatistics_CollectsCorrectly()
{
    var collector = new FusionStatisticsCollector();

    collector.RecordOperation(CreateOperation("Add"));
    collector.RecordOperation(CreateOperation("Mul"));
    collector.RecordFusedGroup(new[] { CreateOperation("ReLU"), CreateOperation("Sigmoid") }, FusionPatternType.ElementWise);
    collector.RecordRejection("Layout mismatch");

    var stats = collector.GetCurrentStatistics();

    Assert.AreEqual(4, stats.TotalOperations);
    Assert.AreEqual(2, stats.FusedOperations);
    Assert.AreEqual(1, stats.FusedGroups);
    Assert.AreEqual(1, stats.RejectedFusions);
}
```

## Success Criteria
- Fusible and NoFusion attributes work correctly
- GraphOptions controls fusion behavior globally
- FusionContext properly manages option scope
- Fusion statistics track operations accurately
- Extension methods provide convenient API
- All APIs are thread-safe where appropriate

## Dependencies
- Operation and FusedOperation types
- FusionPatternType enum
- FusionEngine (for applying fusion)
- ILogger (for logging)
