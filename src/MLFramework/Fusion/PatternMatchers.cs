using MLFramework.Core;

namespace MLFramework.Fusion;

/// <summary>
/// Static class containing pattern matching strategies for common fusion patterns
/// </summary>
public static class PatternMatchers
{
    /// <summary>
    /// Matches element-wise operation chains (e.g., Add -> Mul -> ReLU)
    /// </summary>
    public static bool MatchElementWiseChain(IReadOnlyList<Operation> operations)
    {
        if (operations.Count < 2)
            return false;

        // All operations must be element-wise
        foreach (var op in operations)
        {
            if (!IsElementWiseOperation(op.Type))
                return false;
        }

        // Check shape compatibility
        for (int i = 1; i < operations.Count; i++)
        {
            if (!ShapesCompatible(operations[i - 1], operations[i]))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Matches Conv2D followed by activation (e.g., Conv2D -> ReLU)
    /// </summary>
    public static bool MatchConvActivation(IReadOnlyList<Operation> operations)
    {
        if (operations.Count != 2)
            return false;

        return operations[0].Type == "Conv2D" &&
               IsActivationOperation(operations[1].Type) &&
               ConvActivationCompatible(operations[0], operations[1]);
    }

    /// <summary>
    /// Matches Conv2D followed by BatchNorm for parameter folding
    /// </summary>
    public static bool MatchConvBatchNorm(IReadOnlyList<Operation> operations)
    {
        if (operations.Count != 2)
            return false;

        return operations[0].Type == "Conv2D" &&
               operations[1].Type == "BatchNorm" &&
               BatchNormFoldable(operations[1]);
    }

    /// <summary>
    /// Matches Linear followed by activation (e.g., Linear -> ReLU)
    /// </summary>
    public static bool MatchLinearActivation(IReadOnlyList<Operation> operations)
    {
        if (operations.Count != 2)
            return false;

        return operations[0].Type == "Linear" &&
               IsActivationOperation(operations[1].Type);
    }

    /// <summary>
    /// Determines if an operation type is element-wise
    /// </summary>
    private static bool IsElementWiseOperation(string opType)
    {
        return opType is "Add" or "Sub" or "Mul" or "Div" or
               "ReLU" or "Sigmoid" or "Tanh" or "LeakyReLU" or
               "Exp" or "Log" or "Abs" or "Neg";
    }

    /// <summary>
    /// Determines if an operation type is an activation function
    /// </summary>
    private static bool IsActivationOperation(string opType)
    {
        return opType is "ReLU" or "Sigmoid" or "Tanh" or "LeakyReLU" or
               "GELU" or "Swish" or "Softmax";
    }

    /// <summary>
    /// Checks if two operations have compatible shapes for element-wise operations
    /// </summary>
    private static bool ShapesCompatible(Operation op1, Operation op2)
    {
        // Element-wise ops should maintain same shape
        return op1.OutputShape.IsCompatibleWith(op2.InputShape);
    }

    /// <summary>
    /// Checks if a convolution and activation are compatible for fusion
    /// </summary>
    private static bool ConvActivationCompatible(Operation conv, Operation activation)
    {
        // Check that the conv output shape matches activation input shape
        return conv.OutputShape.IsCompatibleWith(activation.InputShape);
    }

    /// <summary>
    /// Checks if a BatchNorm operation is foldable (typically inference mode)
    /// </summary>
    private static bool BatchNormFoldable(Operation bn)
    {
        // Check if BatchNorm is in inference mode (no running stats)
        if (bn.Attributes.TryGetValue("training", out var trainingObj) &&
            trainingObj is bool training)
        {
            return !training;
        }

        // Default to foldable for inference
        return true;
    }
}
