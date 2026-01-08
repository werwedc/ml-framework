namespace RitterFramework.Core;

/// <summary>
/// Interface for operation metadata providing validation and shape inference capabilities.
/// </summary>
public interface IOperationMetadata
{
    /// <summary>
    /// Gets the type of this operation.
    /// </summary>
    OperationType Type { get; }

    /// <summary>
    /// Gets the human-readable name of this operation.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the number of input tensors required by this operation.
    /// </summary>
    int RequiredInputTensors { get; }

    /// <summary>
    /// Validates that the input shapes are compatible with this operation.
    /// </summary>
    /// <param name="inputShapes">Array of input shapes to validate.</param>
    /// <returns>A ValidationResult indicating whether validation succeeded.</returns>
    bool ValidateInputShapes(params long[][] inputShapes);

    /// <summary>
    /// Infers the output shape given the input shapes for this operation.
    /// </summary>
    /// <param name="inputShapes">Array of input shapes.</param>
    /// <returns>The inferred output shape.</returns>
    long[] InferOutputShape(params long[][] inputShapes);
}
