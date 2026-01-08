namespace MLFramework.HAL.CUDA.Graphs.Attributes;

/// <summary>
/// Attribute to enable automatic CUDA graph capture for methods
/// </summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public class CaptureGraphAttribute : Attribute
{
    /// <summary>
    /// Gets or sets the name of the graph
    /// </summary>
    public string GraphName { get; set; }

    /// <summary>
    /// Gets or sets the number of warm-up iterations before capture
    /// </summary>
    public int WarmupIterations { get; set; } = 3;

    /// <summary>
    /// Gets or sets whether to enable weight updates
    /// </summary>
    public bool EnableWeightUpdates { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable automatic fallback
    /// </summary>
    public bool EnableFallback { get; set; } = true;

    /// <summary>
    /// Creates a new CaptureGraphAttribute with default settings
    /// </summary>
    public CaptureGraphAttribute()
    {
        GraphName = string.Empty;
    }

    /// <summary>
    /// Creates a new CaptureGraphAttribute with the specified graph name
    /// </summary>
    /// <param name="graphName">The name of the graph to capture</param>
    public CaptureGraphAttribute(string graphName)
    {
        GraphName = graphName ?? throw new ArgumentNullException(nameof(graphName));
    }
}
