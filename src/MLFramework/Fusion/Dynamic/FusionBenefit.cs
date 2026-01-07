namespace MLFramework.Fusion.Dynamic;

/// <summary>
/// Represents the estimated benefits of fusing operations
/// </summary>
public class FusionBenefit
{
    /// <summary>
    /// Gets the estimated speedup factor (e.g., 1.5 for 50% speedup)
    /// </summary>
    public required double EstimatedSpeedup { get; init; }

    /// <summary>
    /// Gets the estimated memory saved in bytes
    /// </summary>
    public required long MemorySaved { get; init; }

    /// <summary>
    /// Gets the reduction in kernel count
    /// </summary>
    public required int KernelCountReduction { get; init; }

    /// <summary>
    /// Gets a complexity score (lower is better)
    /// </summary>
    public required double ComplexityScore { get; init; }

    /// <summary>
    /// Determines whether fusion should be performed based on a benefit threshold
    /// </summary>
    /// <param name="threshold">The minimum threshold for benefit (default 1.1 for 10% speedup)</param>
    /// <returns>True if fusion should be performed; otherwise, false</returns>
    public bool ShouldFuse(double threshold = 1.1)
    {
        return EstimatedSpeedup >= threshold && KernelCountReduction > 0;
    }

    /// <summary>
    /// Creates a zero-benefit instance (no benefit from fusion)
    /// </summary>
    public static FusionBenefit None() => new FusionBenefit
    {
        EstimatedSpeedup = 1.0,
        MemorySaved = 0,
        KernelCountReduction = 0,
        ComplexityScore = double.MaxValue
    };

    /// <summary>
    /// Creates a benefit instance with the specified values
    /// </summary>
    public static FusionBenefit Create(double speedup, long memorySaved, int kernelReduction, double complexity)
    {
        return new FusionBenefit
        {
            EstimatedSpeedup = speedup,
            MemorySaved = memorySaved,
            KernelCountReduction = kernelReduction,
            ComplexityScore = complexity
        };
    }
}
