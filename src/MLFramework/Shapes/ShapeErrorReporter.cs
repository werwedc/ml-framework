namespace MLFramework.Shapes;

/// <summary>
/// Provides user-friendly formatting and suggestions for shape mismatch errors.
/// </summary>
public class ShapeErrorReporter
{
    /// <summary>
    /// Formats a ShapeMismatchException into a user-friendly message.
    /// </summary>
    /// <param name="ex">The exception to format.</param>
    /// <returns>A formatted error message.</returns>
    /// <exception cref="ArgumentNullException">Thrown when ex is null.</exception>
    public string FormatError(ShapeMismatchException ex)
    {
        if (ex == null)
            throw new ArgumentNullException(nameof(ex));

        var sb = new System.Text.StringBuilder();

        sb.AppendLine("=== Shape Mismatch Error ===");
        sb.AppendLine();
        sb.AppendLine($"Operation: {ex.OperationName}");
        sb.AppendLine();

        if (ex.ExpectedShapes.Count > 0)
        {
            sb.AppendLine("Expected Shapes:");
            foreach (var shape in ex.ExpectedShapes)
            {
                sb.AppendLine($"  {shape}");
            }
            sb.AppendLine();
        }

        if (ex.ActualShapes.Count > 0)
        {
            sb.AppendLine("Actual Shapes:");
            foreach (var shape in ex.ActualShapes)
            {
                sb.AppendLine($"  {shape}");
            }
            sb.AppendLine();
        }

        if (!string.IsNullOrWhiteSpace(ex.Details))
        {
            sb.AppendLine("Details:");
            sb.AppendLine($"  {ex.Details}");
            sb.AppendLine();
        }

        var suggestion = SuggestFix(ex);
        if (!string.IsNullOrWhiteSpace(suggestion))
        {
            sb.AppendLine("Suggestion:");
            sb.AppendLine($"  {suggestion}");
        }

        return sb.ToString().TrimEnd();
    }

    /// <summary>
    /// Provides a hint on how to resolve the shape mismatch.
    /// </summary>
    /// <param name="ex">The exception to analyze.</param>
    /// <returns>A suggested fix or hint.</returns>
    /// <exception cref="ArgumentNullException">Thrown when ex is null.</exception>
    public string SuggestFix(ShapeMismatchException ex)
    {
        if (ex == null)
            throw new ArgumentNullException(nameof(ex));

        var opName = ex.OperationName.ToLowerInvariant();

        // Operation-specific suggestions
        if (opName.Contains("matmul") || opName.Contains("matmulcompatibility"))
        {
            return "For matrix multiplication, ensure the inner dimensions match. " +
                   "Check that the last dimension of the first tensor equals the second-to-last dimension of the second tensor.";
        }

        if (opName.Contains("reshape"))
        {
            return "For reshape operations, ensure the total number of elements is preserved. " +
                   "You can use -1 as one dimension to let it be inferred automatically.";
        }

        if (opName.Contains("broadcast") || opName.Contains("broadcastcompatibility"))
        {
            return "For broadcasting, ensure dimensions are compatible (equal, one, or missing). " +
                   "Use .unsqueeze() or .expand() to make shapes compatible.";
        }

        if (opName.Contains("sequence"))
        {
            return "Check the operation sequence and ensure output tensor names are properly registered. " +
                   "Verify that each operation's input tensors are produced by previous operations.";
        }

        // General shape mismatch suggestions
        if (ex.ExpectedShapes.Count > 0 && ex.ActualShapes.Count > 0)
        {
            var expectedRank = ex.ExpectedShapes[0].Rank;
            var actualRank = ex.ActualShapes[0].Rank;

            if (expectedRank != actualRank)
            {
                return $"Rank mismatch: expected {expectedRank} dimensions, got {actualRank}. " +
                       $"Consider using .unsqueeze() or .squeeze() to adjust the rank.";
            }

            // Check for specific dimension mismatches
            for (int i = 0; i < Math.Min(expectedRank, actualRank); i++)
            {
                var expectedDim = ex.ExpectedShapes[0].GetDimension(i);
                var actualDim = ex.ActualShapes[0].GetDimension(i);

                if (expectedDim.IsKnown() && actualDim.IsKnown() &&
                    expectedDim.Value != actualDim.Value)
                {
                    if (actualDim.Value == 1)
                    {
                        return $"Dimension {i} is 1 but should be {expectedDim.Value}. " +
                               $"Consider using .repeat() or .expand() to replicate the tensor.";
                    }

                    if (expectedDim.Value == 1)
                    {
                        return $"Dimension {i} is {actualDim.Value} but should be 1. " +
                               $"Consider using .squeeze() to remove this dimension.";
                    }

                    return $"Dimension {i} mismatch: expected {expectedDim.Value}, got {actualDim.Value}. " +
                           $"Check if you need to transpose, permute, or reshape your tensor.";
                }
            }
        }

        return "Review the shapes and operation requirements. Consider using shape inspection " +
               "methods like .shape or printing intermediate shapes during debugging.";
    }

    /// <summary>
    /// Creates an ASCII visualization of the given shapes.
    /// </summary>
    /// <param name="shapes">The shapes to visualize.</param>
    /// <returns>An ASCII representation of the shapes.</returns>
    /// <exception cref="ArgumentNullException">Thrown when shapes is null.</exception>
    public string VisualizeShapes(List<SymbolicShape> shapes)
    {
        if (shapes == null)
            throw new ArgumentNullException(nameof(shapes));

        if (shapes.Count == 0)
        {
            return "No shapes to visualize.";
        }

        var sb = new System.Text.StringBuilder();

        sb.AppendLine("Shape Visualization:");
        sb.AppendLine();

        for (int i = 0; i < shapes.Count; i++)
        {
            sb.AppendLine($"Shape {i + 1}: {shapes[i]}");

            // Create a simple ASCII representation
            VisualizeShapeAscii(shapes[i], sb);
            sb.AppendLine();
        }

        return sb.ToString().TrimEnd();
    }

    /// <summary>
    /// Creates an ASCII representation of a single shape.
    /// </summary>
    /// <param name="shape">The shape to visualize.</param>
    /// <param name="sb">The string builder to append to.</param>
    private void VisualizeShapeAscii(SymbolicShape shape, System.Text.StringBuilder sb)
    {
        if (shape.Rank == 0)
        {
            sb.AppendLine("  Scalar (0D)");
            return;
        }

        if (shape.Rank == 1)
        {
            sb.AppendLine("  1D Tensor: ──────────");
            return;
        }

        if (shape.Rank == 2)
        {
            var rows = shape.GetDimension(0).IsKnown() ? Math.Min(shape.GetDimension(0).Value!.Value, 5) : 3;
            var cols = shape.GetDimension(1).IsKnown() ? Math.Min(shape.GetDimension(1).Value!.Value, 8) : 4;

            sb.AppendLine("  2D Tensor:");
            for (int r = 0; r < rows; r++)
            {
                sb.Append("    ");
                for (int c = 0; c < cols; c++)
                {
                    sb.Append("█ ");
                }
                if (cols < 4 || !shape.GetDimension(1).IsKnown())
                {
                    sb.Append("...");
                }
                sb.AppendLine();
            }
            if (rows < 3 || !shape.GetDimension(0).IsKnown())
            {
                sb.Append("    ...");
                for (int c = 0; c < cols; c++)
                {
                    sb.Append("█ ");
                }
                sb.AppendLine();
            }
            return;
        }

        if (shape.Rank == 3)
        {
            sb.AppendLine("  3D Tensor (batch × height × width):");
            sb.AppendLine("    ┌───────┐");
            sb.AppendLine("    │ █ █ █ │");
            sb.AppendLine("    │ █ █ █ │");
            sb.AppendLine("    └───────┘");
            return;
        }

        // For higher dimensions, show a simplified representation
        sb.AppendLine($"  {shape.Rank}D Tensor: [");
        for (int i = 0; i < Math.Min(shape.Rank, 3); i++)
        {
            var dim = shape.GetDimension(i);
            sb.AppendLine($"    dim {i}: {dim}");
        }
        if (shape.Rank > 3)
        {
            sb.AppendLine($"    ... ({shape.Rank - 3} more dimensions)");
        }
        sb.AppendLine("  ]");
    }

    /// <summary>
    /// Creates a comparison visualization of two shapes.
    /// </summary>
    /// <param name="shapeA">The first shape.</param>
    /// <param name="shapeB">The second shape.</param>
    /// <returns>A visual comparison of the shapes.</returns>
    /// <exception cref="ArgumentNullException">Thrown when shapeA or shapeB is null.</exception>
    public string CompareShapes(SymbolicShape shapeA, SymbolicShape shapeB)
    {
        if (shapeA == null)
            throw new ArgumentNullException(nameof(shapeA));

        if (shapeB == null)
            throw new ArgumentNullException(nameof(shapeB));

        var sb = new System.Text.StringBuilder();

        sb.AppendLine("Shape Comparison:");
        sb.AppendLine();

        var maxRank = Math.Max(shapeA.Rank, shapeB.Rank);

        sb.AppendLine("  Index | Shape A            | Shape B            | Match?");
        sb.AppendLine("  ------|--------------------|--------------------|--------");

        for (int i = 0; i < maxRank; i++)
        {
            var dimA = shapeA.Rank > i ? shapeA.GetDimension(i) : null;
            var dimB = shapeB.Rank > i ? shapeB.GetDimension(i) : null;

            var strA = dimA?.ToString() ?? "N/A";
            var strB = dimB?.ToString() ?? "N/A";

            string match;
            if (dimA == null || dimB == null)
            {
                match = "-";
            }
            else if (dimA.IsKnown() && dimB.IsKnown())
            {
                match = (dimA.Value == dimB.Value || dimA.Value == 1 || dimB.Value == 1) ? "✓" : "✗";
            }
            else
            {
                // For symbolic dimensions, check bounds overlap
                var overlap = !(dimA.MinValue > dimB.MaxValue || dimB.MinValue > dimA.MaxValue);
                match = overlap ? "~" : "✗";
            }

            sb.AppendLine($"  {i,5} | {strA,-18} | {strB,-18} | {match}");
        }

        return sb.ToString().TrimEnd();
    }
}
