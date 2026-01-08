namespace MLFramework.Compilation;

/// <summary>
/// Represents a shape signature for cache lookup
/// </summary>
public readonly struct ShapeSignature : IEquatable<ShapeSignature>
{
    /// <summary>
    /// Gets the operation name
    /// </summary>
    public required string OperationName { get; init; }

    /// <summary>
    /// Gets the input shapes
    /// </summary>
    public required int[][] InputShapes { get; init; }

    /// <summary>
    /// Gets the precomputed hash for fast lookup
    /// </summary>
    public int Hash { get; private init; }

    /// <summary>
    /// Creates a shape signature from an operation and shapes
    /// </summary>
    public static ShapeSignature Create(string operationName, List<int[]> shapes)
    {
        var inputShapes = shapes.Select(s => s.ToArray()).ToArray();
        return new ShapeSignature
        {
            OperationName = operationName,
            InputShapes = inputShapes,
            Hash = ComputeHash(operationName, inputShapes)
        };
    }

    /// <summary>
    /// Computes a hash from operation name and input shapes
    /// </summary>
    private static int ComputeHash(string operationName, int[][] inputShapes)
    {
        unchecked
        {
            int hash = 17;
            hash = hash * 31 + operationName.GetHashCode();

            foreach (var shape in inputShapes)
            {
                foreach (int dim in shape)
                {
                    hash = hash * 31 + dim.GetHashCode();
                }
            }

            return hash;
        }
    }

    /// <summary>
    /// Determines equality between two shape signatures
    /// </summary>
    public bool Equals(ShapeSignature other)
    {
        if (OperationName != other.OperationName)
            return false;

        if (InputShapes.Length != other.InputShapes.Length)
            return false;

        for (int i = 0; i < InputShapes.Length; i++)
        {
            if (!InputShapes[i].SequenceEqual(other.InputShapes[i]))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Determines equality between two objects
    /// </summary>
    public override bool Equals(object? obj)
    {
        return obj is ShapeSignature other && Equals(other);
    }

    /// <summary>
    /// Gets the hash code
    /// </summary>
    public override int GetHashCode() => Hash;

    /// <summary>
    /// Returns a string representation of the shape signature
    /// </summary>
    public override string ToString()
    {
        var shapesStr = string.Join(", ", InputShapes.Select(s => $"[{string.Join(", ", s)}]"));
        return $"{OperationName}({shapesStr})";
    }

    /// <summary>
    /// Equality operator
    /// </summary>
    public static bool operator ==(ShapeSignature left, ShapeSignature right)
    {
        return left.Equals(right);
    }

    /// <summary>
    /// Inequality operator
    /// </summary>
    public static bool operator !=(ShapeSignature left, ShapeSignature right)
    {
        return !(left == right);
    }
}
