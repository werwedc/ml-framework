using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Provides validation utilities for gradient accumulation operations.
/// Ensures shape compatibility and detects potential issues in accumulation.
/// </summary>
public static class GradientAccumulationValidator
{
    /// <summary>
    /// Validates that a list of gradients and batch sizes are compatible for accumulation.
    /// Checks that all gradients have the same shape and all batch sizes are valid.
    /// </summary>
    /// <param name="gradients">List of gradient tensors to validate.</param>
    /// <param name="batchSizes">List of batch sizes corresponding to each gradient.</param>
    /// <returns>True if all validations pass, false otherwise.</returns>
    /// <exception cref="ArgumentNullException">Thrown when gradients or batchSizes is null.</exception>
    public static bool ValidateAccumulation(List<Tensor> gradients, List<int> batchSizes)
    {
        if (gradients == null)
            throw new ArgumentNullException(nameof(gradients));

        if (batchSizes == null)
            throw new ArgumentNullException(nameof(batchSizes));

        // Check if lists have the same length
        if (gradients.Count != batchSizes.Count)
            return false;

        // Check if lists are empty
        if (gradients.Count == 0)
            return false;

        // Check first gradient as reference
        if (gradients[0] == null)
            return false;

        var referenceShape = gradients[0].Shape;

        // Validate each gradient and batch size
        for (int i = 0; i < gradients.Count; i++)
        {
            // Check gradient is not null
            if (gradients[i] == null)
                return false;

            // Check batch size is valid
            if (batchSizes[i] < 1)
                return false;

            // Check shape compatibility
            if (!CheckShapeCompatibility(gradients[0], gradients[i]))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Checks if two gradient tensors have compatible shapes for accumulation.
    /// Shapes are compatible if they have the same dimensions and sizes.
    /// </summary>
    /// <param name="grad1">First gradient tensor.</param>
    /// <param name="grad2">Second gradient tensor.</param>
    /// <returns>True if shapes are compatible, false otherwise.</returns>
    /// <exception cref="ArgumentNullException">Thrown when grad1 or grad2 is null.</exception>
    public static bool CheckShapeCompatibility(Tensor grad1, Tensor grad2)
    {
        if (grad1 == null)
            throw new ArgumentNullException(nameof(grad1));

        if (grad2 == null)
            throw new ArgumentNullException(nameof(grad2));

        // Check if shapes have the same rank
        if (grad1.Shape.Length != grad2.Shape.Length)
            return false;

        // Check each dimension
        for (int i = 0; i < grad1.Shape.Length; i++)
        {
            if (grad1.Shape[i] != grad2.Shape[i])
                return false;
        }

        return true;
    }

    /// <summary>
    /// Validates that an accumulated gradient tensor is compatible with its component gradients.
    /// </summary>
    /// <param name="accumulated">The accumulated gradient tensor.</param>
    /// <param name="components">List of component gradient tensors.</param>
    /// <returns>True if the accumulated shape is valid, false otherwise.</returns>
    /// <exception cref="ArgumentNullException">Thrown when accumulated or components is null.</exception>
    public static bool ValidateAccumulatedShape(Tensor accumulated, List<Tensor> components)
    {
        if (accumulated == null)
            throw new ArgumentNullException(nameof(accumulated));

        if (components == null)
            throw new ArgumentNullException(nameof(components));

        // Check if components list is empty
        if (components.Count == 0)
            return false;

        // Use first component as reference
        if (components[0] == null)
            return false;

        // Check if accumulated shape matches component shape
        return CheckShapeCompatibility(accumulated, components[0]);
    }

    /// <summary>
    /// Validates that a batch size is valid for gradient operations.
    /// </summary>
    /// <param name="batchSize">The batch size to validate.</param>
    /// <returns>True if batch size is valid, false otherwise.</returns>
    public static bool IsValidBatchSize(int batchSize)
    {
        return batchSize > 0;
    }

    /// <summary>
    /// Validates that a target batch size is valid for accumulation.
    /// </summary>
    /// <param name="targetBatchSize">The target batch size to validate.</param>
    /// <returns>True if target batch size is valid, false otherwise.</returns>
    public static bool IsValidTargetBatchSize(int targetBatchSize)
    {
        // 0 is valid for fixed step count mode
        return targetBatchSize >= 0;
    }

    /// <summary>
    /// Validates that accumulation progress is within valid bounds [0.0, 1.0].
    /// </summary>
    /// <param name="progress">The progress value to validate.</param>
    /// <returns>True if progress is valid, false otherwise.</returns>
    public static bool IsValidProgress(double progress)
    {
        return progress >= 0.0 && progress <= 1.0;
    }

    /// <summary>
    /// Validates that buffer indices are valid for the given buffer size.
    /// </summary>
    /// <param name="startIdx">The starting index.</param>
    /// <param name="count">The count of elements.</param>
    /// <param name="bufferSize">The total buffer size.</param>
    /// <returns>True if indices are valid, false otherwise.</returns>
    public static bool ValidateBufferIndices(int startIdx, int count, int bufferSize)
    {
        if (startIdx < 0)
            return false;

        if (count < 0)
            return false;

        if (startIdx + count > bufferSize)
            return false;

        return true;
    }

    /// <summary>
    /// Checks if an accumulation operation would overflow the target batch size.
    /// </summary>
    /// <param name="currentBatchSize">Current accumulated batch size.</param>
    /// <param name="additionalBatchSize">Additional batch size to add.</param>
    /// <param name="targetBatchSize">The target batch size.</param>
    /// <returns>True if overflow would occur, false otherwise.</returns>
    public static bool CheckOverflow(int currentBatchSize, int additionalBatchSize, int targetBatchSize)
    {
        if (targetBatchSize == 0)
            return false; // No target, no overflow

        long newTotal = (long)currentBatchSize + additionalBatchSize;
        return newTotal > targetBatchSize;
    }

    /// <summary>
    /// Validates that a gradient tensor contains valid values.
    /// Checks for NaN and infinity values.
    /// </summary>
    /// <param name="gradient">The gradient tensor to validate.</param>
    /// <returns>True if gradient contains valid values, false otherwise.</returns>
    /// <exception cref="ArgumentNullException">Thrown when gradient is null.</exception>
    public static bool ValidateGradientValues(Tensor gradient)
    {
        if (gradient == null)
            throw new ArgumentNullException(nameof(gradient));

        if (gradient.Data == null)
            return false;

        foreach (var value in gradient.Data)
        {
            if (float.IsNaN(value) || float.IsInfinity(value))
                return false;
        }

        return true;
    }
}
