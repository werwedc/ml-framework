namespace MLFramework.Utilities;

using System.Collections.Generic;

/// <summary>
/// Known non-deterministic operations categorization
/// </summary>
public static class KnownNonDeterministicOperations
{
    /// <summary>
    /// Operations that cannot be made deterministic
    /// </summary>
    public static readonly IReadOnlySet<string> NonDeterminizable = new HashSet<string>
    {
        "atomic_add",
        "atomic_sub",
        "scatter_add",
        "scatter_sub",
        "parallel_sort",  // Some sorting algorithms are non-deterministic
        "hashmap_lookup"  // Hash-based operations
    };

    /// <summary>
    /// Operations that can be made deterministic with performance impact
    /// </summary>
    public static readonly IReadOnlySet<string> Determinizable = new HashSet<string>
    {
        "convolution",
        "matmul",
        "dropout",
        "batch_norm",
        "shuffle",
        "random_sample"
    };
}
