namespace MLFramework.Memory;

/// <summary>
/// Adaptive padding strategy that grows allocations when needed and
/// shrinks them when utilization is consistently low.
/// </summary>
public class AdaptivePaddingStrategy : IPaddingStrategy
{
    /// <summary>
    /// Factor by which to grow allocations (default: 1.5).
    /// </summary>
    public double GrowthFactor { get; set; }

    /// <summary>
    /// Threshold below which to consider shrinking (default: 0.5).
    /// </summary>
    public double ShrinkThreshold { get; set; }

    /// <summary>
    /// Creates a new adaptive padding strategy.
    /// </summary>
    public AdaptivePaddingStrategy(double growthFactor = 1.5, double shrinkThreshold = 0.5)
    {
        if (growthFactor <= 1.0)
        {
            throw new ArgumentException("GrowthFactor must be greater than 1.0", nameof(growthFactor));
        }
        if (shrinkThreshold <= 0.0 || shrinkThreshold >= 1.0)
        {
            throw new ArgumentException("ShrinkThreshold must be between 0 and 1 (exclusive)", nameof(shrinkThreshold));
        }

        GrowthFactor = growthFactor;
        ShrinkThreshold = shrinkThreshold;
    }

    /// <summary>
    /// Calculates the required allocation size with adaptive padding.
    /// </summary>
    public long CalculateRequiredSize(ShapeBounds bounds, int elementSize)
    {
        long expectedElements = bounds.CalculateExpectedElements();
        long maxElements = bounds.CalculateMaxElements();

        // Start with expected size, apply growth factor
        long requiredElements = (long)(expectedElements * GrowthFactor);

        // Cap at max shape size to avoid excessive allocation
        if (requiredElements > maxElements)
        {
            requiredElements = maxElements;
        }

        return requiredElements * elementSize;
    }

    /// <summary>
    /// Determines if a memory handle should be resized for a new shape.
    /// </summary>
    public bool ShouldResize(IMemoryHandle handle, int[] newShape)
    {
        long newRequiredSize = handle.ShapeBounds.CalculateElements(newShape) * 4; // Assuming float32 for now

        // Always resize if we need more space
        if (newRequiredSize > handle.CapacityBytes)
        {
            return true;
        }

        // Check if we should shrink based on utilization
        double utilization = (double)newRequiredSize / handle.CapacityBytes;
        if (utilization < ShrinkThreshold)
        {
            // Only shrink if the new shape is significantly smaller than expected
            // to avoid thrashing
            long expectedSize = handle.ShapeBounds.CalculateExpectedElements() * 4;
            if (newRequiredSize < expectedSize * 0.75)
            {
                return true;
            }
        }

        return false;
    }
}
