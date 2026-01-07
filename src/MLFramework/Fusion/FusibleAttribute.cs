namespace MLFramework.Fusion;

/// <summary>
/// Attribute to mark methods or operations as explicitly fusible
/// </summary>
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
