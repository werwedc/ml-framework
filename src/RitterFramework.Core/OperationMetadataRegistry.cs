namespace RitterFramework.Core;

/// <summary>
/// Centralized registry for operation metadata and validation.
/// </summary>
public class OperationMetadataRegistry
{
    private static readonly Lazy<OperationMetadataRegistry> _instance =
        new Lazy<OperationMetadataRegistry>(() => new OperationMetadataRegistry());

    private readonly Dictionary<OperationType, IOperationMetadata> _metadataMap;

    /// <summary>
    /// Gets the singleton instance of the registry.
    /// </summary>
    public static OperationMetadataRegistry Instance => _instance.Value;

    /// <summary>
    /// Private constructor for singleton pattern.
    /// </summary>
    private OperationMetadataRegistry()
    {
        _metadataMap = new Dictionary<OperationType, IOperationMetadata>();
        RegisterDefaultOperations();
    }

    /// <summary>
    /// Registers operation metadata for a specific operation type.
    /// </summary>
    /// <param name="type">The operation type to register.</param>
    /// <param name="metadata">The metadata implementation.</param>
    public void Register(OperationType type, IOperationMetadata metadata)
    {
        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata));
        }

        if (type != metadata.Type)
        {
            throw new ArgumentException($"Metadata type {metadata.Type} does not match registration type {type}");
        }

        _metadataMap[type] = metadata;
    }

    /// <summary>
    /// Gets metadata for a specific operation type.
    /// </summary>
    /// <param name="type">The operation type.</param>
    /// <returns>The operation metadata.</returns>
    /// <exception cref="KeyNotFoundException">Thrown when the operation type is not registered.</exception>
    public IOperationMetadata GetMetadata(OperationType type)
    {
        if (!_metadataMap.TryGetValue(type, out var metadata))
        {
            throw new KeyNotFoundException($"Operation type '{type}' is not registered in the metadata registry.");
        }

        return metadata;
    }

    /// <summary>
    /// Validates operation inputs using the registered metadata.
    /// </summary>
    /// <param name="type">The operation type.</param>
    /// <param name="inputShapes">Array of input shapes to validate.</param>
    /// <returns>A ValidationResult indicating whether validation succeeded.</returns>
    public ValidationResult Validate(OperationType type, params long[][] inputShapes)
    {
        if (inputShapes == null || inputShapes.Length == 0)
        {
            return ValidationResult.Invalid("Input shapes array cannot be null or empty");
        }

        try
        {
            var metadata = GetMetadata(type);

            if (inputShapes.Length != metadata.RequiredInputTensors)
            {
                return ValidationResult.Invalid(
                    $"Operation '{metadata.Name}' requires {metadata.RequiredInputTensors} input tensors, but {inputShapes.Length} were provided");
            }

            bool isValid = metadata.ValidateInputShapes(inputShapes);

            if (!isValid)
            {
                return ValidationResult.Invalid(
                    $"Input shapes are not compatible with operation '{metadata.Name}'");
            }

            return ValidationResult.Valid();
        }
        catch (Exception ex)
        {
            return ValidationResult.Invalid($"Validation failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Infers the output shape for an operation given input shapes.
    /// </summary>
    /// <param name="type">The operation type.</param>
    /// <param name="inputShapes">Array of input shapes.</param>
    /// <returns>The inferred output shape.</returns>
    public long[] InferOutputShape(OperationType type, params long[][] inputShapes)
    {
        var metadata = GetMetadata(type);
        return metadata.InferOutputShape(inputShapes);
    }

    /// <summary>
    /// Checks if an operation type is registered.
    /// </summary>
    /// <param name="type">The operation type to check.</param>
    /// <returns>True if the operation type is registered, false otherwise.</returns>
    public bool IsRegistered(OperationType type)
    {
        return _metadataMap.ContainsKey(type);
    }

    /// <summary>
    /// Registers default operation metadata implementations.
    /// </summary>
    private void RegisterDefaultOperations()
    {
        // Register basic operations
        Register(OperationType.MatrixMultiply, new Operations.MatrixMultiplyMetadata());
        Register(OperationType.Conv2D, new Operations.Conv2DMetadata());
        Register(OperationType.Concat, new Operations.ConcatMetadata());
    }
}
