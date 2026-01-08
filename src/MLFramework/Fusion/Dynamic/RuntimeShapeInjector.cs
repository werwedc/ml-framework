using MLFramework.Core;

namespace MLFramework.Fusion.Dynamic;

/// <summary>
/// Injects runtime shape checks and dispatch logic into fusion nodes
/// </summary>
public class RuntimeShapeInjector
{
    /// <summary>
    /// Injects runtime shape validation into a fusion node
    /// </summary>
    /// <param name="node">The fusion node to inject shape checks into</param>
    /// <returns>A list of operations including shape validation</returns>
    public List<Operation> InjectShapeCheck(FusionNode node)
    {
        if (node == null)
            return new List<Operation>();

        var operations = new List<Operation>();

        // Add shape validation operation at the beginning
        var shapeCheckOp = CreateShapeCheckOperation(node);
        operations.Add(shapeCheckOp);

        // Add the original operations
        foreach (var op in node.Operations)
        {
            operations.Add(op);
        }

        return operations;
    }

    /// <summary>
    /// Generates a dispatch operation that selects the specialized kernel based on runtime shape
    /// </summary>
    /// <param name="node">The fusion node to generate dispatch for</param>
    /// <returns>A dispatch operation</returns>
    public Operation GenerateShapeDispatch(FusionNode node)
    {
        if (node == null)
            throw new ArgumentNullException(nameof(node));

        var signature = node.GetFusedSignature();

        return new Operation
        {
            Id = $"dispatch_{node.FusionId}",
            Type = "ShapeDispatch",
            Name = $"ShapeDispatch_{node.FusionId}",
            DataType = DataType.Float32,
            Layout = TensorLayout.NCHW,
            InputShape = new TensorShape { Dimensions = new[] { 0 } }, // Placeholder
            OutputShape = new TensorShape { Dimensions = new[] { 0 } }, // Placeholder
            Inputs = node.Operations.SelectMany(op => op.Inputs).ToList(),
            Outputs = node.Operations.SelectMany(op => op.Outputs).ToList(),
            Attributes = new Dictionary<string, object>
            {
                ["FusionId"] = node.FusionId,
                ["Signature"] = signature,
                ["DispatchType"] = "ShapeBased"
            }
        };
    }

    /// <summary>
    /// Generates a generic fallback operation for unknown shapes
    /// </summary>
    /// <param name="node">The fusion node to generate fallback for</param>
    /// <returns>A generic fallback operation</returns>
    public Operation GenerateGenericFallback(FusionNode node)
    {
        if (node == null)
            throw new ArgumentNullException(nameof(node));

        return new Operation
        {
            Id = $"generic_fallback_{node.FusionId}",
            Type = "GenericFusedKernel",
            Name = $"GenericFusedKernel_{node.FusionId}",
            DataType = DataType.Float32,
            Layout = TensorLayout.NCHW,
            InputShape = new TensorShape { Dimensions = new[] { 0 } }, // Placeholder
            OutputShape = new TensorShape { Dimensions = new[] { 0 } }, // Placeholder
            Inputs = node.Operations.SelectMany(op => op.Inputs).ToList(),
            Outputs = node.Operations.SelectMany(op => op.Outputs).ToList(),
            Attributes = new Dictionary<string, object>
            {
                ["FusionId"] = node.FusionId,
                ["OperationCount"] = node.Operations.Count,
                ["KernelType"] = "Generic"
            }
        };
    }

    /// <summary>
    /// Creates a shape check operation for the fusion node
    /// </summary>
    private Operation CreateShapeCheckOperation(FusionNode node)
    {
        var expectedShapes = node.InputShapes.Select(s => s.ToString()).ToList();

        return new Operation
        {
            Id = $"shape_check_{node.FusionId}",
            Type = "ShapeValidation",
            Name = $"ShapeValidation_{node.FusionId}",
            DataType = DataType.Float32,
            Layout = TensorLayout.NCHW,
            InputShape = new TensorShape { Dimensions = new[] { 0 } }, // Placeholder
            OutputShape = new TensorShape { Dimensions = new[] { 0 } }, // Placeholder
            Inputs = node.Operations.SelectMany(op => op.Inputs).ToList(),
            Outputs = node.Operations.SelectMany(op => op.Outputs).ToList(),
            Attributes = new Dictionary<string, object>
            {
                ["ExpectedInputShapes"] = expectedShapes,
                ["FusionId"] = node.FusionId
            }
        };
    }
}
