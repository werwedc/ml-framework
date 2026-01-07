namespace MLFramework.Memory;

/// <summary>
/// Represents bounds for dynamic tensor shapes.
/// </summary>
public class ShapeBounds
{
    /// <summary>
    /// Minimum allowed shape for each dimension.
    /// </summary>
    public int[] MinShape { get; }

    /// <summary>
    /// Maximum allowed shape for each dimension.
    /// </summary>
    public int[] MaxShape { get; }

    /// <summary>
    /// Expected shape for allocation decisions.
    /// </summary>
    public int[] ExpectedShape { get; }

    /// <summary>
    /// Creates a new shape bounds instance.
    /// </summary>
    public ShapeBounds(int[] minShape, int[] maxShape, int[] expectedShape)
    {
        if (minShape.Length != maxShape.Length || minShape.Length != expectedShape.Length)
        {
            throw new ArgumentException("All shape arrays must have the same length.");
        }

        for (int i = 0; i < minShape.Length; i++)
        {
            if (minShape[i] < 0)
            {
                throw new ArgumentException($"MinShape dimension {i} cannot be negative.");
            }
            if (maxShape[i] < minShape[i])
            {
                throw new ArgumentException($"MaxShape dimension {i} cannot be less than MinShape.");
            }
            if (expectedShape[i] < minShape[i] || expectedShape[i] > maxShape[i])
            {
                throw new ArgumentException($"ExpectedShape dimension {i} must be within MinShape and MaxShape bounds.");
            }
        }

        MinShape = minShape;
        MaxShape = maxShape;
        ExpectedShape = expectedShape;
    }

    /// <summary>
    /// Calculates the maximum number of elements this bounds can hold.
    /// </summary>
    public long CalculateMaxElements()
    {
        long elements = 1;
        foreach (int dim in MaxShape)
        {
            elements *= dim;
        }
        return elements;
    }

    /// <summary>
    /// Calculates the expected number of elements based on expected shape.
    /// </summary>
    public long CalculateExpectedElements()
    {
        long elements = 1;
        foreach (int dim in ExpectedShape)
        {
            elements *= dim;
        }
        return elements;
    }

    /// <summary>
    /// Calculates the number of elements for a specific shape.
    /// </summary>
    public long CalculateElements(int[] shape)
    {
        if (shape.Length != MinShape.Length)
        {
            throw new ArgumentException("Shape must have the same rank as bounds.");
        }

        long elements = 1;
        foreach (int dim in shape)
        {
            elements *= dim;
        }
        return elements;
    }

    /// <summary>
    /// Checks if a given shape is within the bounds.
    /// </summary>
    public bool Contains(int[] shape)
    {
        if (shape.Length != MinShape.Length)
        {
            return false;
        }

        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] < MinShape[i] || shape[i] > MaxShape[i])
            {
                return false;
            }
        }

        return true;
    }

    public override string ToString()
    {
        var minStr = string.Join("x", MinShape);
        var maxStr = string.Join("x", MaxShape);
        var expectedStr = string.Join("x", ExpectedShape);
        return $"[{minStr} -> {maxStr}] (expected: {expectedStr})";
    }
}
