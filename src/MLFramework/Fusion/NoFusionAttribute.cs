namespace MLFramework.Fusion;

/// <summary>
/// Attribute to prevent fusion of specific operations or methods
/// </summary>
[AttributeUsage(AttributeTargets.Method | AttributeTargets.Class, AllowMultiple = false)]
public class NoFusionAttribute : Attribute
{
    /// <summary>
    /// Creates a NoFusion attribute with default reason
    /// </summary>
    public NoFusionAttribute()
    {
        Reason = "Explicitly marked as non-fusible";
    }

    /// <summary>
    /// Creates a NoFusion attribute with custom reason
    /// </summary>
    /// <param name="reason">Reason for preventing fusion (for logging/debugging)</param>
    public NoFusionAttribute(string reason)
    {
        Reason = reason;
    }

    /// <summary>
    /// Reason for preventing fusion (for logging/debugging)
    /// </summary>
    public string Reason { get; set; } = "Explicitly marked as non-fusible";
}
