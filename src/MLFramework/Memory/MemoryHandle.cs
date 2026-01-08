using System;

namespace MLFramework.Memory;

/// <summary>
/// Implementation of IMemoryHandle for tracking allocated memory.
/// </summary>
public class MemoryHandle : IMemoryHandle
{
    /// <summary>
    /// Pointer to the allocated memory.
    /// </summary>
    public IntPtr Pointer { get; private set; }

    /// <summary>
    /// Total capacity in bytes.
    /// </summary>
    public long CapacityBytes { get; private set; }

    /// <summary>
    /// Current shape of the tensor.
    /// </summary>
    public int[] CurrentShape { get; private set; }

    /// <summary>
    /// Shape bounds for this allocation.
    /// </summary>
    public ShapeBounds ShapeBounds { get; }

    /// <summary>
    /// Timestamp when this allocation was created.
    /// </summary>
    public DateTime AllocationTime { get; }

    /// <summary>
    /// Element size in bytes.
    /// </summary>
    private readonly int _elementSize;

    /// <summary>
    /// Callback for when memory needs to be reallocated.
    /// </summary>
    private readonly Action<IntPtr, long> _resizeCallback;

    /// <summary>
    /// Creates a new memory handle.
    /// </summary>
    public MemoryHandle(
        IntPtr pointer,
        long capacityBytes,
        int[] initialShape,
        ShapeBounds bounds,
        int elementSize,
        Action<IntPtr, long>? resizeCallback = null)
    {
        Pointer = pointer;
        CapacityBytes = capacityBytes;
        CurrentShape = initialShape;
        ShapeBounds = bounds;
        AllocationTime = DateTime.UtcNow;
        _elementSize = elementSize;
        _resizeCallback = resizeCallback ?? ((ptr, size) => { });
    }

    /// <summary>
    /// Resizes the memory allocation to a new shape.
    /// </summary>
    public void Resize(int[] newShape)
    {
        if (!ShapeBounds.Contains(newShape))
        {
            throw new ArgumentException($"Shape {string.Join("x", newShape)} is outside bounds {ShapeBounds}");
        }

        long newRequiredBytes = ShapeBounds.CalculateElements(newShape) * _elementSize;

        if (newRequiredBytes > CapacityBytes)
        {
            // Need to resize the underlying allocation
            _resizeCallback(Pointer, newRequiredBytes);
            CapacityBytes = newRequiredBytes;
        }

        CurrentShape = newShape;
    }

    /// <summary>
    /// Calculates the effective size in bytes for the current shape.
    /// </summary>
    public long GetEffectiveSize()
    {
        return ShapeBounds.CalculateElements(CurrentShape) * _elementSize;
    }

    /// <summary>
    /// Calculates the current utilization ratio (effective size / capacity).
    /// </summary>
    public double GetUtilization()
    {
        return CapacityBytes > 0 ? (double)GetEffectiveSize() / CapacityBytes : 0.0;
    }
}
