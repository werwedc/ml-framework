using System;
using System.Linq;

namespace RitterFramework.Core.Operations;

/// <summary>
/// Metadata for concatenation operations.
/// Validates that all input tensors have the same shape except on the concatenation axis.
/// </summary>
public class ConcatMetadata : IOperationMetadata
{
    private readonly int _axis;

    /// <summary>
    /// Creates a ConcatMetadata that concatenates along the first axis (default).
    /// </summary>
    public ConcatMetadata() : this(axis: 0)
    {
    }

    /// <summary>
    /// Creates a ConcatMetadata that concatenates along the specified axis.
    /// </summary>
    /// <param name="axis">The axis along which to concatenate.</param>
    public ConcatMetadata(int axis)
    {
        if (axis < 0)
        {
            throw new ArgumentException("Axis cannot be negative", nameof(axis));
        }

        _axis = axis;
    }

    /// <inheritdoc/>
    public OperationType Type => OperationType.Concat;

    /// <inheritdoc/>
    public string Name => $"Concat (axis={_axis})";

    /// <inheritdoc/>
    public int RequiredInputTensors => 2;

    /// <inheritdoc/>
    public bool ValidateInputShapes(params long[][] inputShapes)
    {
        if (inputShapes.Length < 2)
        {
            return false;
        }

        // Get the reference shape (first tensor)
        var referenceShape = inputShapes[0];

        // All tensors must have the same rank
        if (inputShapes.Any(shape => shape.Length != referenceShape.Length))
        {
            return false;
        }

        // All dimensions must match except on the concatenation axis
        for (int i = 0; i < referenceShape.Length; i++)
        {
            if (i == _axis)
            {
                // Skip the concatenation axis (dimensions can differ)
                continue;
            }

            if (inputShapes.Any(shape => shape[i] != referenceShape[i]))
            {
                return false;
            }
        }

        // Check if axis is valid
        if (_axis >= referenceShape.Length)
        {
            return false;
        }

        return true;
    }

    /// <inheritdoc/>
    public long[] InferOutputShape(params long[][] inputShapes)
    {
        var referenceShape = inputShapes[0];
        var outputShape = new long[referenceShape.Length];

        // Copy all dimensions from reference
        Array.Copy(referenceShape, outputShape, referenceShape.Length);

        // Sum the dimensions on the concatenation axis
        long concatDim = inputShapes.Sum(shape => shape[_axis]);
        outputShape[_axis] = concatDim;

        return outputShape;
    }

    /// <summary>
    /// Gets the concatenation axis.
    /// </summary>
    public int Axis => _axis;
}
