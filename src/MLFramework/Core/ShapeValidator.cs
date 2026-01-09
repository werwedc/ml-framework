using RitterFramework.Core;

namespace MLFramework.Core;

/// <summary>
/// Provides shape validation logic for various tensor operations.
/// </summary>
public static class ShapeValidator
{
    /// <summary>
    /// Validates shapes before matrix multiplication.
    /// Shape1: [M, K], Shape2: [K, N]
    /// </summary>
    /// <param name="shape1">First tensor shape.</param>
    /// <param name="shape2">Second tensor shape.</param>
    /// <returns>Validation result indicating if shapes are compatible for matrix multiplication.</returns>
    public static ValidationResult ValidateMatrixMultiply(long[] shape1, long[] shape2)
    {
        if (shape1 == null)
            throw new ArgumentNullException(nameof(shape1));
        if (shape2 == null)
            throw new ArgumentNullException(nameof(shape2));

        if (shape1.Length != 2)
            return ValidationResult.Invalid(
                $"First tensor must be 2-dimensional for matrix multiplication, got shape: [{string.Join(", ", shape1)}]");

        if (shape2.Length != 2)
            return ValidationResult.Invalid(
                $"Second tensor must be 2-dimensional for matrix multiplication, got shape: [{string.Join(", ", shape2)}]");

        // Inner dimensions must match
        if (shape1[1] != shape2[0])
        {
            var errorMessage = $"Matrix multiplication: Shape [{string.Join(", ", shape1)}] × [{string.Join(", ", shape2)}] invalid. " +
                              $"Inner dimensions {shape1[1]} and {shape2[0]} must match.";
            var suggestions = new List<string>
            {
                "Check layer configurations",
                $"Previous layer outputs {shape1[1]} features, but this layer expects {shape2[0]}",
                "Adjust input/output dimensions in layer configuration"
            };
            return ValidationResult.Invalid(errorMessage, suggestions);
        }

        return ValidationResult.Valid();
    }

    /// <summary>
    /// Validates shapes before 2D convolution.
    /// Input: [N, C, H, W], Kernel: [F, C, kH, kW]
    /// </summary>
    /// <param name="inputShape">Input tensor shape.</param>
    /// <param name="kernelShape">Kernel tensor shape.</param>
    /// <param name="stride">Stride value for convolution.</param>
    /// <param name="padding">Padding value for convolution.</param>
    /// <returns>Validation result indicating if shapes are compatible for convolution.</returns>
    public static ValidationResult ValidateConv2D(
        long[] inputShape,
        long[] kernelShape,
        int stride,
        int padding)
    {
        if (inputShape == null)
            throw new ArgumentNullException(nameof(inputShape));
        if (kernelShape == null)
            throw new ArgumentNullException(nameof(kernelShape));

        if (inputShape.Length != 4)
            return ValidationResult.Invalid(
                $"Input must be 4-dimensional [N, C, H, W] for Conv2D, got shape: [{string.Join(", ", inputShape)}]");

        if (kernelShape.Length != 4)
            return ValidationResult.Invalid(
                $"Kernel must be 4-dimensional [F, C, kH, kW] for Conv2D, got shape: [{string.Join(", ", kernelShape)}]");

        // Input channels must match kernel channels
        if (inputShape[1] != kernelShape[1])
        {
            var errorMessage = $"Conv2D: Input [{string.Join(", ", inputShape)}] with kernel [{string.Join(", ", kernelShape)}]. " +
                              $"Input channels ({inputShape[1]}) do not match kernel channels ({kernelShape[1]})";
            var suggestions = new List<string>
            {
                "Check channel configurations",
                "Verify input tensor has correct channel dimension",
                $"Ensure input has {kernelShape[1]} channels or adjust kernel channels"
            };
            return ValidationResult.Invalid(errorMessage, suggestions);
        }

        // Validate kernel dimensions don't exceed input dimensions
        if (kernelShape[2] > inputShape[2] || kernelShape[3] > inputShape[3])
        {
            var errorMessage = $"Conv2D: Kernel size [{kernelShape[2]}, {kernelShape[3]}] exceeds input spatial dimensions [{inputShape[2]}, {inputShape[3]}]";
            var suggestions = new List<string>
            {
                "Reduce kernel size",
                "Increase input spatial dimensions",
                "Consider using padding to accommodate kernel size"
            };
            return ValidationResult.Invalid(errorMessage, suggestions);
        }

        // Validate stride and padding
        if (stride < 1)
            return ValidationResult.Invalid($"Stride must be at least 1, got {stride}");

        if (padding < 0)
            return ValidationResult.Invalid($"Padding cannot be negative, got {padding}");

        return ValidationResult.Valid();
    }

    /// <summary>
    /// Validates shapes before concatenation.
    /// All shapes must match except on the concatenation axis.
    /// </summary>
    /// <param name="inputShapes">List of input tensor shapes.</param>
    /// <param name="axis">Axis along which to concatenate.</param>
    /// <returns>Validation result indicating if shapes are compatible for concatenation.</returns>
    public static ValidationResult ValidateConcat(List<long[]> inputShapes, int axis)
    {
        if (inputShapes == null || inputShapes.Count == 0)
            return ValidationResult.Invalid("Input shapes list cannot be empty");

        var firstShape = inputShapes[0];

        // Validate axis is within bounds
        if (axis < 0 || axis >= firstShape.Length)
            return ValidationResult.Invalid(
                $"Concatenation axis {axis} is out of bounds for tensors with {firstShape.Length} dimensions");

        // Validate all inputs have same number of dimensions
        for (int i = 0; i < inputShapes.Count; i++)
        {
            if (inputShapes[i] == null)
                return ValidationResult.Invalid($"Input shape at index {i} is null");

            if (inputShapes[i].Length != firstShape.Length)
                return ValidationResult.Invalid(
                    $"Cannot concatenate tensors with different dimensions: " +
                    $"tensor 0 has {firstShape.Length} dims, tensor {i} has {inputShapes[i].Length} dims");
        }

        // Validate all dimensions match except on concatenation axis
        for (int dim = 0; dim < firstShape.Length; dim++)
        {
            if (dim == axis) continue;

            for (int i = 1; i < inputShapes.Count; i++)
            {
                if (inputShapes[i][dim] != firstShape[dim])
                {
                    var errorMessage = $"Concat on axis {axis}: Dimension {dim} mismatch across inputs. " +
                                      $"Expected {firstShape[dim]}, got {inputShapes[i][dim]} at tensor {i}";
                    var suggestions = new List<string>
                    {
                        $"All dimensions must match except on concatenation axis {axis}",
                        "Check tensor shapes before concatenation",
                        "Consider reshaping inputs or using different concatenation axis"
                    };
                    return ValidationResult.Invalid(errorMessage, suggestions);
                }
            }
        }

        return ValidationResult.Valid();
    }

    /// <summary>
    /// Validates shapes are compatible for broadcasting.
    /// Two shapes are compatible if:
    /// - They are equal, or
    /// - One dimension is 1, or
    /// - One dimension doesn't exist
    /// </summary>
    /// <param name="shape1">First tensor shape.</param>
    /// <param name="shape2">Second tensor shape.</param>
    /// <returns>Validation result indicating if shapes are compatible for broadcasting.</returns>
    public static ValidationResult ValidateBroadcast(long[] shape1, long[] shape2)
    {
        if (shape1 == null)
            throw new ArgumentNullException(nameof(shape1));
        if (shape2 == null)
            throw new ArgumentNullException(nameof(shape2));

        // Start from the rightmost dimensions and work left
        int len1 = shape1.Length;
        int len2 = shape2.Length;
        int maxLen = Math.Max(len1, len2);

        for (int i = 1; i <= maxLen; i++)
        {
            // Get dimensions from the right
            long dim1 = i <= len1 ? shape1[len1 - i] : 1;
            long dim2 = i <= len2 ? shape2[len2 - i] : 1;

            // Check compatibility
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
            {
                var errorMessage = $"Broadcast: [{string.Join(", ", shape1)}] and [{string.Join(", ", shape2)}] are not compatible. " +
                                  $"Cannot broadcast dimension {dim1} with {dim2}";
                var suggestions = new List<string>
                {
                    "Ensure dimensions are equal or one of them is 1",
                    "Consider reshaping tensors to compatible dimensions",
                    "Use explicit expansion instead of broadcasting"
                };
                return ValidationResult.Invalid(errorMessage, suggestions);
            }
        }

        return ValidationResult.Valid();
    }

    /// <summary>
    /// Generic validation using operation type.
    /// </summary>
    /// <param name="operationType">Type of operation to validate.</param>
    /// <param name="inputShapes">Input tensor shapes.</param>
    /// <returns>Validation result for the specified operation.</returns>
    public static ValidationResult Validate(OperationType operationType, params long[][] inputShapes)
    {
        if (inputShapes == null || inputShapes.Length == 0)
            return ValidationResult.Invalid("Input shapes array cannot be empty");

        return operationType switch
        {
            OperationType.MatrixMultiply or OperationType.Linear => ValidateMatrixMultiply(inputShapes[0], inputShapes[1]),
            OperationType.Conv2D => inputShapes.Length >= 2
                ? ValidateConv2D(inputShapes[0], inputShapes[1], stride: 1, padding: 0)
                : ValidationResult.Invalid("Conv2D requires at least 2 input shapes (input and kernel)"),
            OperationType.Concat => inputShapes.Length >= 2
                ? ValidateConcat(inputShapes.ToList(), axis: 0)
                : ValidationResult.Invalid("Concat requires at least 2 input shapes and an axis parameter"),
            OperationType.Broadcast => inputShapes.Length >= 2
                ? ValidateBroadcast(inputShapes[0], inputShapes[1])
                : ValidationResult.Invalid("Broadcast requires at least 2 input shapes"),
            _ => ValidationResult.Valid() // Other operations don't require specific validation
        };
    }
}
