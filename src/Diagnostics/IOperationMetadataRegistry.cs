using System.Collections.Generic;

namespace MLFramework.Diagnostics
{
    /// <summary>
    /// Interface for a registry that stores shape requirements and validation rules for operations.
    /// </summary>
    public interface IOperationMetadataRegistry
    {
        /// <summary>
        /// Registers shape requirements for an operation type.
        /// </summary>
        /// <param name="operationType">The type of operation to register.</param>
        /// <param name="requirements">The shape requirements for the operation.</param>
        void RegisterOperation(
            MLFramework.Core.OperationType operationType,
            OperationShapeRequirements requirements);

        /// <summary>
        /// Gets the shape requirements for a registered operation.
        /// </summary>
        /// <param name="operationType">The type of operation to get requirements for.</param>
        /// <returns>The shape requirements for the operation, or null if not registered.</returns>
        OperationShapeRequirements GetRequirements(MLFramework.Core.OperationType operationType);

        /// <summary>
        /// Checks if an operation type is registered in the registry.
        /// </summary>
        /// <param name="operationType">The type of operation to check.</param>
        /// <returns>True if the operation is registered, false otherwise.</returns>
        bool IsRegistered(MLFramework.Core.OperationType operationType);

        /// <summary>
        /// Validates input tensor shapes against an operation's requirements.
        /// </summary>
        /// <param name="operationType">The type of operation to validate against.</param>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <param name="operationParameters">Optional operation parameters used in validation.</param>
        /// <returns>A ValidationResult indicating whether the shapes are valid.</returns>
        ValidationResult ValidateShapes(
            MLFramework.Core.OperationType operationType,
            IEnumerable<long[]> inputShapes,
            System.Collections.IDictionary<string, object> operationParameters = null);
    }
}
