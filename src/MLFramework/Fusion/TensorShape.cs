namespace MLFramework.Fusion;

/// <summary>
/// Represents tensor shape
/// </summary>
public record TensorShape
{
    /// <summary>
    /// Gets the dimensions of the tensor
    /// </summary>
    public required IReadOnlyList<int> Dimensions { get; init; }

    /// <summary>
    /// Gets the total number of elements in the tensor
    /// </summary>
    public int Size => Dimensions.Count > 0
        ? Dimensions.Aggregate(1, (acc, dim) => acc * dim)
        : 0;

    /// <summary>
    /// Gets the number of dimensions (rank)
    /// </summary>
    public int Rank => Dimensions.Count;

    /// <summary>
    /// Gets the total number of elements in the tensor (alias for Size)
    /// </summary>
    public int TotalElements => Size;

    /// <summary>
    /// Gets the last dimension (width for 2D/4D tensors)
    /// </summary>
    public int Width => Dimensions.Count > 0 ? Dimensions[^1] : 1;

    /// <summary>
    /// Gets the second-to-last dimension (height for 4D tensors)
    /// </summary>
    public int Height => Dimensions.Count > 1 ? Dimensions[^2] : 1;

    /// <summary>
    /// Creates a tensor shape from the given dimensions
    /// </summary>
    public static TensorShape Create(params int[] dimensions)
    {
        return new TensorShape { Dimensions = dimensions };
    }

    /// <summary>
    /// Checks if this shape is compatible with another shape for element-wise operations
    /// </summary>
    public bool IsCompatibleWith(TensorShape other)
    {
        if (Dimensions.Count != other.Dimensions.Count)
            return false;

        return Dimensions.SequenceEqual(other.Dimensions);
    }

    /// <summary>
    /// Returns a string representation of the tensor shape
    /// </summary>
    public override string ToString()
    {
        return $"[{string.Join(", ", Dimensions)}]";
    }
}
