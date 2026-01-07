using System;

namespace MLFramework.Conversion;

/// <summary>
/// Represents analysis result for a single layer in tensor parallelism context.
/// </summary>
public class LayerAnalysisResult
{
    /// <summary>
    /// Gets or sets the name of the layer.
    /// </summary>
    public string LayerName { get; set; } = "";

    /// <summary>
    /// Gets or sets the type of the layer (e.g., Linear, Conv2d).
    /// </summary>
    public string LayerType { get; set; } = "";

    /// <summary>
    /// Gets or sets a value indicating whether this layer is parallelizable.
    /// </summary>
    public bool IsParallelizable { get; set; }

    /// <summary>
    /// Gets or sets the suggested parallelism type for this layer.
    /// </summary>
    public ParallelismType SuggestedParallelism { get; set; }

    /// <summary>
    /// Gets or sets the memory usage in bytes for this layer.
    /// </summary>
    public long MemoryBytes { get; set; }

    /// <summary>
    /// Gets or sets the number of parameters in this layer.
    /// </summary>
    public long ParameterCount { get; set; }

    /// <summary>
    /// Creates a result indicating the layer is not parallelizable.
    /// </summary>
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

    /// <summary>
    /// Creates a result indicating the layer should use column parallelism.
    /// </summary>
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

    /// <summary>
    /// Creates a result indicating the layer should use row parallelism.
    /// </summary>
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

/// <summary>
/// Defines the types of parallelism that can be applied to a layer.
/// </summary>
public enum ParallelismType
{
    /// <summary>
    /// No parallelism applied.
    /// </summary>
    None,

    /// <summary>
    /// Column parallelism - parallelize along output dimension.
    /// </summary>
    Column,

    /// <summary>
    /// Row parallelism - parallelize along input dimension.
    /// </summary>
    Row,

    /// <summary>
    /// Convolution output channel parallelism.
    /// </summary>
    ConvOutput,

    /// <summary>
    /// Convolution input channel parallelism.
    /// </summary>
    ConvInput
}
