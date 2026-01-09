namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Interface for a central registry that stores shape requirements and validation rules for operations.
/// </summary>
public interface IOperationMetadataRegistry
{
    /// <summary>
    /// Register shape requirements for an operation.
    /// </summary>
    void RegisterOperation(
        OperationType operationType,
        OperationShapeRequirements requirements);

    /// <summary>
    /// Get shape requirements for an operation.
    /// </summary>
    OperationShapeRequirements GetRequirements(OperationType operationType);

    /// <summary>
    /// Check if operation is registered.
    /// </summary>
    bool IsRegistered(OperationType operationType);

    /// <summary>
    /// Validate shapes against operation requirements.
    /// </summary>
    ValidationResult ValidateShapes(
        OperationType operationType,
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters = null);
}
