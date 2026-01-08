using System.Linq;
using MLFramework.Fusion;
using MLFramework.Shapes.Inference;

namespace MLFramework.Shapes;

/// <summary>
/// Performs static shape checking to detect shape incompatibilities before execution.
/// Uses aggressive checking to prefer false positives over runtime errors.
/// </summary>
public class StaticShapeChecker
{
    private readonly ShapeInferenceEngine _inferenceEngine;

    /// <summary>
    /// Initializes a new instance of the StaticShapeChecker class with a default inference engine.
    /// </summary>
    public StaticShapeChecker()
        : this(new ShapeInferenceEngine())
    {
    }

    /// <summary>
    /// Initializes a new instance of the StaticShapeChecker class with a custom inference engine.
    /// </summary>
    /// <param name="inferenceEngine">The shape inference engine to use.</param>
    /// <exception cref="ArgumentNullException">Thrown when inferenceEngine is null.</exception>
    public StaticShapeChecker(ShapeInferenceEngine inferenceEngine)
    {
        _inferenceEngine = inferenceEngine ?? throw new ArgumentNullException(nameof(inferenceEngine));
    }

    /// <summary>
    /// Checks if an operation's input shapes are valid.
    /// Throws a ShapeMismatchException if the shapes are invalid.
    /// </summary>
    /// <param name="op">The operation to check.</param>
    /// <param name="inputs">The input shapes for the operation.</param>
    /// <exception cref="ArgumentNullException">Thrown when op or inputs is null.</exception>
    /// <exception cref="ShapeMismatchException">Thrown when the operation's input shapes are invalid.</exception>
    public void CheckOperation(Operation op, List<SymbolicShape> inputs)
    {
        if (op == null)
            throw new ArgumentNullException(nameof(op));

        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        // Check if the operation type is registered
        if (!_inferenceEngine.HasRule(op.Type))
        {
            throw new ShapeMismatchException(
                op.Type,
                new List<SymbolicShape>(),
                inputs,
                $"Operation type '{op.Type}' is not registered for shape inference.");
        }

        // Check if the operation can be inferred with these inputs
        if (!_inferenceEngine.CanInfer(op.Type, inputs))
        {
            throw new ShapeMismatchException(
                op.Type,
                new List<SymbolicShape>(),
                inputs,
                $"Cannot infer output shapes for operation '{op.Type}' with {inputs.Count} input(s).");
        }
    }

    /// <summary>
    /// Checks a sequence of operations for shape compatibility.
    /// Accumulates all errors before throwing.
    /// </summary>
    /// <param name="ops">The sequence of operations to check.</param>
    /// <param name="tensorShapes">A dictionary mapping tensor names to their shapes.</param>
    /// <exception cref="ArgumentNullException">Thrown when ops or tensorShapes is null.</exception>
    /// <exception cref="ShapeMismatchException">Thrown when any operation in the sequence has invalid shapes.</exception>
    public void CheckSequence(List<Operation> ops, Dictionary<string, SymbolicShape> tensorShapes)
    {
        if (ops == null)
            throw new ArgumentNullException(nameof(ops));

        if (tensorShapes == null)
            throw new ArgumentNullException(nameof(tensorShapes));

        var errors = new List<ShapeMismatchException>();

        foreach (var op in ops)
        {
            try
            {
                // Gather input shapes
                var inputShapes = new List<SymbolicShape>();
                foreach (var inputTensor in op.Inputs)
                {
                    if (!tensorShapes.TryGetValue(inputTensor, out var shape))
                    {
                        throw new ShapeMismatchException(
                            op.Type,
                            new List<SymbolicShape>(),
                            new List<SymbolicShape>(),
                            $"Input tensor '{inputTensor}' not found in shape registry.");
                    }
                    inputShapes.Add(shape);
                }

                // Check the operation
                CheckOperation(op, inputShapes);

                // Infer and register output shapes
                var outputShapes = _inferenceEngine.Infer(op.Type, inputShapes);
                for (int i = 0; i < op.Outputs.Count && i < outputShapes.Count; i++)
                {
                    tensorShapes[op.Outputs[i]] = outputShapes[i];
                }
            }
            catch (ShapeMismatchException ex)
            {
                errors.Add(ex);
            }
        }

        if (errors.Count > 0)
        {
            throw new ShapeMismatchException(
                "Sequence",
                errors.SelectMany(e => e.ExpectedShapes).ToList(),
                errors.SelectMany(e => e.ActualShapes).ToList(),
                $"Found {errors.Count} shape error(s) in operation sequence:\n" +
                string.Join("\n", errors.Select((e, i) => $"  {i + 1}. {e.Message}")));
        }
    }

    /// <summary>
    /// Checks if two shapes are compatible for broadcasting.
    /// </summary>
    /// <param name="a">The first shape.</param>
    /// <param name="b">The second shape.</param>
    /// <exception cref="ShapeMismatchException">Thrown when the shapes are not compatible for broadcasting.</exception>
    public void CheckBroadcastCompatibility(SymbolicShape a, SymbolicShape b)
    {
        if (a == null)
            throw new ArgumentNullException(nameof(a));

        if (b == null)
            throw new ArgumentNullException(nameof(b));

        var maxRank = Math.Max(a.Rank, b.Rank);
        var errors = new List<string>();

        // Check each dimension from right to left
        for (int i = 1; i <= maxRank; i++)
        {
            var dimA = a.Rank >= i ? a.GetDimension(-i) : null;
            var dimB = b.Rank >= i ? b.GetDimension(-i) : null;

            // If both dimensions are known and different (and neither is 1), it's an error
            if (dimA != null && dimB != null &&
                dimA.IsKnown() && dimB.IsKnown() &&
                dimA.Value != 1 && dimB.Value != 1 &&
                dimA.Value != dimB.Value)
            {
                errors.Add($"Dimension {-i}: {dimA.Value} vs {dimB.Value}");
            }

            // Use symbolic bounds for checking
            if (dimA != null && dimB != null &&
                !dimA.IsKnown() && !dimB.IsKnown())
            {
                // If both are symbolic, check bounds for incompatibility
                if (dimA.MinValue > dimB.MaxValue || dimB.MinValue > dimA.MaxValue)
                {
                    var dimAMax = dimA.MaxValue?.ToString() ?? "∞";
                    var dimBMax = dimB.MaxValue?.ToString() ?? "∞";
                    errors.Add($"Dimension {-i}: bounds [{dimA.MinValue}..{dimAMax}] vs [{dimB.MinValue}..{dimBMax}]");
                }
            }
        }

        if (errors.Count > 0)
        {
            throw new ShapeMismatchException(
                "Broadcast",
                new List<SymbolicShape> { b },
                new List<SymbolicShape> { a },
                $"Shapes {a} and {b} are not compatible for broadcasting:\n  {string.Join("\n  ", errors)}");
        }
    }

    /// <summary>
    /// Checks if a reshape operation is valid.
    /// </summary>
    /// <param name="from">The original shape.</param>
    /// <param name="to">The target shape.</param>
    /// <exception cref="ShapeMismatchException">Thrown when the reshape is not valid.</exception>
    public void CheckReshapeValid(SymbolicShape from, SymbolicShape to)
    {
        if (from == null)
            throw new ArgumentNullException(nameof(from));

        if (to == null)
            throw new ArgumentNullException(nameof(to));

        // If both shapes are fully known, check if total elements match
        if (from.IsFullyKnown() && to.IsFullyKnown())
        {
            var fromProduct = from.Dimensions.Aggregate(1, (acc, dim) => acc * dim.Value!.Value);
            var toProduct = to.Dimensions.Aggregate(1, (acc, dim) => acc * dim.Value!.Value);

            if (fromProduct != toProduct)
            {
                throw new ShapeMismatchException(
                    "Reshape",
                    new List<SymbolicShape> { to },
                    new List<SymbolicShape> { from },
                    $"Cannot reshape {from} to {to}: total element count mismatch ({fromProduct} vs {toProduct}).");
            }
        }
        else
        {
            // Use symbolic bounds for checking
            long fromMin = 1;
            long? fromMax = 1;
            foreach (var dim in from.Dimensions)
            {
                fromMin *= dim.MinValue;
                if (fromMax.HasValue && dim.MaxValue.HasValue)
                {
                    fromMax = fromMax.Value * dim.MaxValue;
                }
                else
                {
                    fromMax = null;
                }
            }

            long toMin = 1;
            long? toMax = 1;
            foreach (var dim in to.Dimensions)
            {
                toMin *= dim.MinValue;
                if (toMax.HasValue && dim.MaxValue.HasValue)
                {
                    toMax = toMax.Value * dim.MaxValue;
                }
                else
                {
                    toMax = null;
                }
            }

            // Check if bounds are compatible
            if (fromMax.HasValue && toMin > fromMax)
            {
                throw new ShapeMismatchException(
                    "Reshape",
                    new List<SymbolicShape> { to },
                    new List<SymbolicShape> { from },
                    $"Cannot reshape {from} to {to}: minimum element count of target shape ({toMin}) exceeds maximum of source shape ({fromMax}).");
            }

            if (toMax.HasValue && fromMin > toMax)
            {
                throw new ShapeMismatchException(
                    "Reshape",
                    new List<SymbolicShape> { to },
                    new List<SymbolicShape> { from },
                    $"Cannot reshape {from} to {to}: minimum element count of source shape ({fromMin}) exceeds maximum of target shape ({toMax}).");
            }
        }
    }

    /// <summary>
    /// Checks if two shapes are compatible for matrix multiplication.
    /// </summary>
    /// <param name="a">The first tensor shape.</param>
    /// <param name="b">The second tensor shape.</param>
    /// <exception cref="ShapeMismatchException">Thrown when the shapes are not compatible for matmul.</exception>
    public void CheckMatMulCompatibility(SymbolicShape a, SymbolicShape b)
    {
        if (a == null)
            throw new ArgumentNullException(nameof(a));

        if (b == null)
            throw new ArgumentNullException(nameof(b));

        // For matrix multiplication, we need at least 2 dimensions
        if (a.Rank < 2)
        {
            throw new ShapeMismatchException(
                "MatMul",
                new List<SymbolicShape> { a },
                new List<SymbolicShape> { a },
                $"Shape {a} has insufficient dimensions for matrix multiplication (rank {a.Rank} < 2).");
        }

        if (b.Rank < 2)
        {
            throw new ShapeMismatchException(
                "MatMul",
                new List<SymbolicShape> { b },
                new List<SymbolicShape> { b },
                $"Shape {b} has insufficient dimensions for matrix multiplication (rank {b.Rank} < 2).");
        }

        // Check if the last dimension of A matches the second-to-last dimension of B
        var aLastDim = a.GetDimension(-1);
        var bSecondLastDim = b.GetDimension(-2);

        if (aLastDim.IsKnown() && bSecondLastDim.IsKnown())
        {
            if (aLastDim.Value != bSecondLastDim.Value)
            {
                throw new ShapeMismatchException(
                    "MatMul",
                    new List<SymbolicShape> { new SymbolicShape(
                        bSecondLastDim,
                        b.GetDimension(-1)) },
                    new List<SymbolicShape> { new SymbolicShape(
                        a.GetDimension(-2),
                        aLastDim) },
                    $"Cannot multiply shapes {a} and {b}: inner dimensions {aLastDim.Value} and {bSecondLastDim.Value} must match.");
            }
        }
        else
        {
            // Use symbolic bounds for checking
            if (aLastDim.MinValue > bSecondLastDim.MaxValue || bSecondLastDim.MinValue > aLastDim.MaxValue)
            {
                throw new ShapeMismatchException(
                    "MatMul",
                    new List<SymbolicShape> { new SymbolicShape(
                        bSecondLastDim,
                        b.GetDimension(-1)) },
                    new List<SymbolicShape> { new SymbolicShape(
                        a.GetDimension(-2),
                        aLastDim) },
                    $"Cannot multiply shapes {a} and {b}: inner dimension bounds are incompatible [{aLastDim.MinValue}..{aLastDim.MaxValue?.ToString() ?? "∞"}] vs [{bSecondLastDim.MinValue}..{bSecondLastDim.MaxValue?.ToString() ?? "∞"}].");
            }
        }

        // Check batch dimension compatibility
        var maxRank = Math.Max(a.Rank, b.Rank);
        for (int i = 3; i <= maxRank; i++)
        {
            var dimA = a.Rank >= i ? a.GetDimension(-i) : null;
            var dimB = b.Rank >= i ? b.GetDimension(-i) : null;

            if (dimA != null && dimB != null &&
                dimA.IsKnown() && dimB.IsKnown() &&
                dimA.Value != 1 && dimB.Value != 1 &&
                dimA.Value != dimB.Value)
            {
                throw new ShapeMismatchException(
                    "MatMul",
                    new List<SymbolicShape> { b },
                    new List<SymbolicShape> { a },
                    $"Batch dimension {-i} incompatible: {dimA.Value} vs {dimB.Value}");
            }
        }
    }

    #region Internal Validation Methods

    /// <summary>
    /// Checks if an operation's inputs have the expected rank.
    /// </summary>
    /// <param name="op">The operation to check.</param>
    /// <param name="inputs">The input shapes.</param>
    /// <param name="expectedRank">The expected rank.</param>
    /// <exception cref="ShapeMismatchException">Thrown when any input does not have the expected rank.</exception>
    internal void CheckRank(Operation op, List<SymbolicShape> inputs, int expectedRank)
    {
        if (op == null)
            throw new ArgumentNullException(nameof(op));

        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        for (int i = 0; i < inputs.Count; i++)
        {
            if (inputs[i].Rank != expectedRank)
            {
                throw new ShapeMismatchException(
                    op.Type,
                    new List<SymbolicShape> { CreateShapeWithRank(expectedRank) },
                    inputs,
                    $"Input {i} has rank {inputs[i].Rank}, expected {expectedRank}.");
            }
        }
    }

    /// <summary>
    /// Checks if a shape has the expected value at a specific dimension.
    /// </summary>
    /// <param name="op">The operation to check.</param>
    /// <param name="shape">The shape to check.</param>
    /// <param name="dimIndex">The dimension index.</param>
    /// <param name="expectedValue">The expected value.</param>
    /// <exception cref="ShapeMismatchException">Thrown when the dimension does not have the expected value.</exception>
    internal void CheckDim(Operation op, SymbolicShape shape, int dimIndex, int expectedValue)
    {
        if (op == null)
            throw new ArgumentNullException(nameof(op));

        if (shape == null)
            throw new ArgumentNullException(nameof(shape));

        var dim = shape.GetDimension(dimIndex);

        if (dim.IsKnown() && dim.Value != expectedValue)
        {
            throw new ShapeMismatchException(
                op.Type,
                new List<SymbolicShape> { CreateShapeWithDimValue(shape.Rank, dimIndex, expectedValue) },
                new List<SymbolicShape> { shape },
                $"Dimension {dimIndex} has value {dim.Value}, expected {expectedValue}.");
        }
        else if (!dim.IsKnown())
        {
            // Use symbolic bounds for checking
            if (dim.MinValue > expectedValue || (dim.MaxValue.HasValue && dim.MaxValue < expectedValue))
            {
                throw new ShapeMismatchException(
                    op.Type,
                    new List<SymbolicShape> { CreateShapeWithDimValue(shape.Rank, dimIndex, expectedValue) },
                    new List<SymbolicShape> { shape },
                    $"Dimension {dimIndex} with bounds [{dim.MinValue}..{dim.MaxValue?.ToString() ?? "∞"}] cannot have value {expectedValue}.");
            }
        }
    }

    /// <summary>
    /// Checks if a dimension's value falls within the expected range.
    /// </summary>
    /// <param name="op">The operation to check.</param>
    /// <param name="shape">The shape to check.</param>
    /// <param name="dimIndex">The dimension index.</param>
    /// <param name="min">The minimum expected value.</param>
    /// <param name="max">The maximum expected value.</param>
    /// <exception cref="ShapeMismatchException">Thrown when the dimension value is outside the range.</exception>
    internal void CheckDimRange(Operation op, SymbolicShape shape, int dimIndex, int min, int max)
    {
        if (op == null)
            throw new ArgumentNullException(nameof(op));

        if (shape == null)
            throw new ArgumentNullException(nameof(shape));

        var dim = shape.GetDimension(dimIndex);

        if (dim.IsKnown())
        {
            if (dim.Value < min || dim.Value > max)
            {
                throw new ShapeMismatchException(
                    op.Type,
                    new List<SymbolicShape> { CreateShapeWithDimValue(shape.Rank, dimIndex, (min + max) / 2) },
                    new List<SymbolicShape> { shape },
                    $"Dimension {dimIndex} has value {dim.Value}, expected range [{min}..{max}].");
            }
        }
        else
        {
            // Use symbolic bounds for checking
            if (dim.MaxValue.HasValue && dim.MaxValue < min)
            {
                throw new ShapeMismatchException(
                    op.Type,
                    new List<SymbolicShape> { CreateShapeWithDimValue(shape.Rank, dimIndex, (min + max) / 2) },
                    new List<SymbolicShape> { shape },
                    $"Dimension {dimIndex} with maximum {dim.MaxValue} is below required minimum {min}.");
            }

            if (dim.MinValue > max)
            {
                throw new ShapeMismatchException(
                    op.Type,
                    new List<SymbolicShape> { CreateShapeWithDimValue(shape.Rank, dimIndex, (min + max) / 2) },
                    new List<SymbolicShape> { shape },
                    $"Dimension {dimIndex} with minimum {dim.MinValue} is above required maximum {max}.");
            }
        }
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Creates a shape with the specified rank for error reporting.
    /// </summary>
    private static SymbolicShape CreateShapeWithRank(int rank)
    {
        var dims = new SymbolicDimension[rank];
        for (int i = 0; i < rank; i++)
        {
            dims[i] = new SymbolicDimension($"dim_{i}");
        }
        return new SymbolicShape(dims);
    }

    /// <summary>
    /// Creates a shape with a specific dimension value for error reporting.
    /// </summary>
    private static SymbolicShape CreateShapeWithDimValue(int rank, int dimIndex, int value)
    {
        var dims = new SymbolicDimension[rank];
        for (int i = 0; i < rank; i++)
        {
            var name = i == dimIndex ? $"{value}" : "dim";
            dims[i] = new SymbolicDimension(name, i == dimIndex ? value : null);
        }
        return new SymbolicShape(dims);
    }

    #endregion
}
