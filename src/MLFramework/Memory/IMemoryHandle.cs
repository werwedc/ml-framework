using System;

namespace MLFramework.Memory;

/// <summary>
/// Interface for a handle to allocated memory.
/// </summary>
public interface IMemoryHandle
{
    /// <summary>
    /// Pointer to the allocated memory.
    /// </summary>
    IntPtr Pointer { get; }

    /// <summary>
    /// Total capacity in bytes.
    /// </summary>
    long CapacityBytes { get; }

    /// <summary>
    /// Current shape of the tensor.
    /// </summary>
    int[] CurrentShape { get; }

    /// <summary>
    /// Shape bounds for this allocation.
    /// </summary>
    ShapeBounds ShapeBounds { get; }

    /// <summary>
    /// Timestamp when this allocation was created.
    /// </summary>
    DateTime AllocationTime { get; }

    /// <summary>
    /// Resizes the memory allocation to a new shape.
    /// </summary>
    void Resize(int[] newShape);

    /// <summary>
    /// Calculates the effective size in bytes for the current shape.
    /// </summary>
    long GetEffectiveSize();

    /// <summary>
    /// Calculates the current utilization ratio (effective size / capacity).
    /// </summary>
    double GetUtilization();
}
