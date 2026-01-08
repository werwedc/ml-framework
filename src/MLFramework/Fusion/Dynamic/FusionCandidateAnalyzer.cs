using MLFramework.Shapes;
using MLFramework.Core;

namespace MLFramework.Fusion.Dynamic;

/// <summary>
/// Analyzes operations to find candidates for fusion
/// </summary>
public class FusionCandidateAnalyzer
{
    private readonly double _minimumBenefitThreshold;

    /// <summary>
    /// Initializes a new instance of the FusionCandidateAnalyzer class
    /// </summary>
    /// <param name="minimumBenefitThreshold">The minimum benefit threshold for fusion (default 1.1)</param>
    public FusionCandidateAnalyzer(double minimumBenefitThreshold = 1.1)
    {
        _minimumBenefitThreshold = minimumBenefitThreshold;
    }

    /// <summary>
    /// Finds fusible operations in a sequence of operations
    /// </summary>
    /// <param name="ops">The list of operations to analyze</param>
    /// <returns>A list of fusion nodes containing fusible operations</returns>
    public List<FusionNode> FindFusibleOperations(List<Operation> ops)
    {
        if (ops == null || ops.Count == 0)
            return new List<FusionNode>();

        var fusionNodes = new List<FusionNode>();
        var visited = new HashSet<int>();

        for (int i = 0; i < ops.Count; i++)
        {
            if (visited.Contains(i))
                continue;

            var fusionNode = new FusionNode { FusionId = Guid.NewGuid().ToString("N") };
            fusionNode.AddOperation(ops[i]);

            // Try to extend the fusion with subsequent operations
            for (int j = i + 1; j < ops.Count; j++)
            {
                if (visited.Contains(j))
                    continue;

                var nextOp = ops[j];

                // Get intermediate shapes (simplified - in reality this would come from shape inference)
                var intermediateShapes = InferIntermediateShapes(ops[i], nextOp);

                if (fusionNode.CanFuseWith(nextOp, intermediateShapes))
                {
                    fusionNode.AddOperation(nextOp);
                    visited.Add(j);
                }
                else
                {
                    // Check if we should start a new fusion from this operation
                    var newBenefit = AnalyzeBenefit(new FusionNode { FusionId = Guid.NewGuid().ToString("N") });
                    if (newBenefit.ShouldFuse(_minimumBenefitThreshold))
                    {
                        break;
                    }
                }
            }

            var benefit = AnalyzeBenefit(fusionNode);
            if (benefit.ShouldFuse(_minimumBenefitThreshold))
            {
                fusionNodes.Add(fusionNode);
                visited.Add(i);
            }
        }

        return fusionNodes;
    }

    /// <summary>
    /// Analyzes the benefit of fusing operations in a fusion node
    /// </summary>
    /// <param name="node">The fusion node to analyze</param>
    /// <returns>The estimated fusion benefit</returns>
    public FusionBenefit AnalyzeBenefit(FusionNode node)
    {
        if (node == null || node.Operations.Count < 2)
            return FusionBenefit.None();

        var operations = node.Operations;
        var kernelCount = operations.Count;

        // Estimate speedup based on operation types and count
        // More operations fused = higher speedup (but diminishing returns)
        var speedup = 1.0 + (kernelCount - 1) * 0.3;
        speedup = Math.Min(speedup, 3.0); // Cap at 3x speedup

        // Estimate memory saved: avoid intermediate outputs
        var memorySaved = 0L;
        for (int i = 0; i < operations.Count - 1; i++)
        {
            memorySaved += EstimateOutputSize(operations[i]);
        }

        // Kernel count reduction = number of operations - 1 (fused into one kernel)
        var kernelReduction = kernelCount - 1;

        // Complexity score: more operations = more complex
        var complexityScore = operations.Count * 1.0;

        return FusionBenefit.Create(speedup, memorySaved, kernelReduction, complexityScore);
    }

    /// <summary>
    /// Determines whether an operation preserves shape (output shape equals input shape)
    /// </summary>
    /// <param name="op">The operation to check</param>
    /// <returns>True if the operation preserves shape; otherwise, false</returns>
    public bool IsShapePreserving(Operation op)
    {
        if (op == null)
            return false;

        // Element-wise operations preserve shape
        var elementWiseOps = new[] { "Add", "Sub", "Mul", "Div", "ReLU", "Sigmoid", "Tanh", "Exp", "Log", "Sqrt", "Pow" };
        if (elementWiseOps.Contains(op.Type, StringComparer.OrdinalIgnoreCase))
            return true;

        // Check if input and output shapes match
        // Note: In a real implementation, this would compare TensorShape or SymbolicShape
        // Here we assume the operation has shape information
        return op.InputShape.Rank == op.OutputShape.Rank &&
               op.InputShape.Dimensions.SequenceEqual(op.OutputShape.Dimensions);
    }

    /// <summary>
    /// Determines whether a fusion node requires runtime shape checking
    /// </summary>
    /// <param name="node">The fusion node to check</param>
    /// <returns>True if runtime shape checks are needed; otherwise, false</returns>
    public bool RequiresRuntimeShapeCheck(FusionNode node)
    {
        if (node == null)
            return false;

        // If any input or output shape has symbolic dimensions, we need runtime checks
        var hasSymbolicInput = node.InputShapes.Any(s => !s.IsFullyKnown());
        var hasSymbolicOutput = node.OutputShapes.Any(s => !s.IsFullyKnown());

        return hasSymbolicInput || hasSymbolicOutput;
    }

    /// <summary>
    /// Infers intermediate shapes between two operations (simplified implementation)
    /// </summary>
    private List<SymbolicShape> InferIntermediateShapes(Operation op1, Operation op2)
    {
        // In a real implementation, this would use a shape inference engine
        // For now, we return empty list to indicate unknown shapes
        return new List<SymbolicShape>();
    }

    /// <summary>
    /// Estimates the output size of an operation in bytes
    /// </summary>
    private long EstimateOutputSize(Operation op)
    {
        if (op == null)
            return 0L;

        // Calculate total elements
        var totalElements = op.OutputShape.Dimensions.Aggregate(1L, (acc, dim) => acc * dim);

        // Assume float32 (4 bytes) by default
        var elementSize = 4L;

        return totalElements * elementSize;
    }
}
