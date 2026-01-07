namespace MLFramework.Fusion;

/// <summary>
/// Fusion strategy for applying transformations
/// </summary>
public enum FusionStrategy
{
    /// <summary>Merge operations into single kernel</summary>
    Merge,
    /// <summary>Fold parameters (e.g., BN into Conv)</summary>
    Fold,
    /// <summary>Replace with specialized kernel</summary>
    Replace,
    /// <summary>Perform operations in-place</summary>
    Inplace
}
