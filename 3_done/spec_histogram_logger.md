# Spec: Histogram Logger

## Overview
Implement a histogram logger for tracking distributions of weights, activations, gradients, and other tensors throughout training.

## Objectives
- Efficient logging of tensor distributions as histograms
- Support multiple histogram types (linear, logarithmic bins)
- Provide statistics (mean, std, min, max, quantiles)
- Enable detection of training issues (dead neurons, exploding gradients)

## API Design

```csharp
// Histogram bin configuration
public class HistogramBinConfig
{
    public int BinCount { get; set; } = 30;
    public bool UseLogScale { get; set; } = false;
    public float Min { get; set; } = float.MinValue;
    public float Max { get; set; } = float.MaxValue;
}

// Histogram data
public class HistogramData
{
    public string Name { get; }
    public long Step { get; }
    public DateTime Timestamp { get; }
    public float[] BinEdges { get; }
    public long[] BinCounts { get; }

    // Statistics
    public float Min { get; }
    public float Max { get; }
    public float Mean { get; }
    public float Std { get; }
    public float[] Quantiles { get; } // [0.1, 0.25, 0.5, 0.75, 0.9]

    public int TotalCount { get; }
    public float BinWidth { get; }
}

// Tensor distribution data
public class TensorDistribution
{
    public string Name { get; }
    public long Step { get; }
    public float[] Values { get; }
    public HistogramData Histogram { get; }

    // Additional statistics
    public float Median { get; }
    public float Skewness { get; }
    public float Kurtosis { get; }
    public int DeadNeuronCount { get; } // Count of zero-valued entries
    public int OutlierCount { get; } // Count > 3 std from mean
}

// Histogram logger interface
public interface IHistogramLogger
{
    void LogHistogram(string name, float[] values, long step = -1);
    void LogHistogram(string name, float[] values, HistogramBinConfig config, long step = -1);
    Task LogHistogramAsync(string name, float[] values, long step = -1);

    void LogDistribution(string name, float[] values, long step = -1);
    Task LogDistributionAsync(string name, float[] values, long step = -1);

    // Retrieval
    HistogramData GetHistogram(string name, long step);
    TensorDistribution GetDistribution(string name, long step);
    IEnumerable<HistogramData> GetHistogramsOverTime(string name);
}

public class HistogramLogger : IHistogramLogger
{
    public HistogramLogger(IStorageBackend storage);
    public HistogramLogger(IEventPublisher eventPublisher);

    // Configuration
    public HistogramBinConfig DefaultBinConfig { get; set; }
    public bool AutoDetectOutliers { get; set; } = true;
    public float OutlierThresholdStd { get; set; } = 3.0f;
}
```

## Implementation Requirements

### 1. HistogramBinConfig and HistogramData (30-45 min)
- Implement bin configuration with customizable parameters
- Implement `HistogramData` class:
  - Compute bin edges (linear or logarithmic)
  - Count values in each bin
  - Compute basic statistics (min, max, mean, std)
  - Compute quantiles (0.1, 0.25, 0.5, 0.75, 0.9)
- Handle edge cases:
  - Empty input array
  - All values the same
  - Very large or very small values

### 2. TensorDistribution (30-45 min)
- Implement `TensorDistribution` with extended statistics:
  - Median (using quickselect or sort)
  - Skewness (third moment)
  - Kurtosis (fourth moment)
  - Dead neuron detection (count zeros)
  - Outlier detection (count values > threshold * std)
- Include computed histogram for visualization
- Add metadata for debugging

### 3. HistogramLogger Core (45-60 min)
- Implement `IHistogramLogger` interface
- Compute histograms efficiently:
  - Use vectorized operations where possible
  - Pre-allocate arrays to avoid allocations
  - Cache bin edges for repeated measurements
- Support automatic bin configuration:
  - Auto-detect min/max from data
  - Auto-select linear or log scale based on data range
- Integrate with event system (publish `HistogramEvent`)
- Integrate with storage backend
- Implement async logging with minimal overhead

### 4. Advanced Statistics (30-45 min)
- Implement efficient quantile computation:
  - Use selection algorithm for median
  - Use interpolation for non-integer positions
- Implement skewness and kurtosis:
  - Use corrected formulas for sample statistics
- Implement outlier detection:
  - Configurable threshold (std multiplier)
  - Report count and optionally values
- Implement dead neuron detection:
  - Count zero-valued entries
  - Optionally count near-zero values

## File Structure
```
src/
  MLFramework.Visualization/
    Histograms/
      HistogramBinConfig.cs
      HistogramData.cs
      TensorDistribution.cs
      IHistogramLogger.cs
      HistogramLogger.cs
      Statistics/
        QuantileCalculator.cs
        MomentsCalculator.cs
        OutlierDetector.cs

tests/
  MLFramework.Visualization.Tests/
    Histograms/
      HistogramLoggerTests.cs
      HistogramDataTests.cs
      StatisticsTests.cs
```

## Dependencies
- `MLFramework.Visualization.Events` (Event system)
- `MLFramework.Visualization.Storage` (Storage backend)

## Integration Points
- Used to track weight/gradient distributions during training
- Integrated with training loop hooks
- Data consumed by tensor distribution visualization

## Success Criteria
- Computing histogram for 1M values completes in <50ms
- Computing all statistics (mean, std, quantiles) for 1M values in <100ms
- Memory usage scales linearly with input size
- Logarithmic bins correctly handle negative values
- Outlier detection accurately identifies anomalies
- Unit tests verify statistical correctness
