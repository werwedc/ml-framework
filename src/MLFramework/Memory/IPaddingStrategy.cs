namespace MLFramework.Memory;

/// <summary>
/// Interface for padding strategies used in dynamic memory allocation.
/// </summary>
public interface IPaddingStrategy
{
    /// <summary>
    /// Calculates the required allocation size for given shape bounds.
    /// </summary>
    long CalculateRequiredSize(ShapeBounds bounds, int elementSize);

    /// <summary>
    /// Determines if a memory handle should be resized for a new shape.
    /// </summary>
    bool ShouldResize(IMemoryHandle handle, int[] newShape);
}
