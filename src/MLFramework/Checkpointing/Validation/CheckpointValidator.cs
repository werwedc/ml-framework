namespace MachineLearning.Checkpointing;

/// <summary>
/// Validator for checkpoint integrity and compatibility
/// </summary>
public class CheckpointValidator : ICheckpointValidator
{
    private readonly ICheckpointStorage _storage;
    private readonly List<IIntegrityChecker> _integrityCheckers;
    private readonly List<ICompatibilityChecker> _compatibilityCheckers;

    /// <summary>
    /// Create a new CheckpointValidator
    /// </summary>
    public CheckpointValidator(
        ICheckpointStorage storage,
        List<IIntegrityChecker>? integrityCheckers = null,
        List<ICompatibilityChecker>? compatibilityCheckers = null)
    {
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
        _integrityCheckers = integrityCheckers ?? new List<IIntegrityChecker>();
        _compatibilityCheckers = compatibilityCheckers ?? new List<ICompatibilityChecker>();
    }

    /// <summary>
    /// Validate a checkpoint
    /// </summary>
    public async Task<ValidationResult> ValidateCheckpointAsync(
        string checkpointPath,
        CancellationToken cancellationToken = default)
    {
        var result = new ValidationResult();

        try
        {
            // Load metadata
            var metadataBytes = await _storage.ReadAsync(checkpointPath, cancellationToken);
            var metadataJson = System.Text.Encoding.UTF8.GetString(metadataBytes);
            var metadata = MetadataSerializer.Deserialize(metadataJson);

            // Validate metadata
            ValidateMetadata(metadata, result);

            // Run integrity checkers
            foreach (var checker in _integrityCheckers)
            {
                try
                {
                    var integrityResult = await checker.CheckFileIntegrityAsync(checkpointPath, cancellationToken);
                    if (!integrityResult.IsValid)
                    {
                        result.AddErrors(integrityResult.Errors);
                    }
                }
                catch (Exception ex)
                {
                    result.AddError($"Integrity checker failed: {ex.Message}");
                }
            }
        }
        catch (Exception ex)
        {
            result.AddError($"Validation failed: {ex.Message}");
        }

        return result;
    }

    /// <summary>
    /// Validate checkpoint metadata
    /// </summary>
    private void ValidateMetadata(CheckpointMetadata metadata, ValidationResult result)
    {
        if (string.IsNullOrWhiteSpace(metadata.Version))
        {
            result.AddError("Missing or invalid version in metadata");
        }

        if (metadata.WorldSize <= 0)
        {
            result.AddError($"Invalid world size: {metadata.WorldSize}");
        }

        if (string.IsNullOrWhiteSpace(metadata.Format))
        {
            result.AddWarning("Missing format in metadata");
        }

        if (metadata.ShardingStrategy == null)
        {
            result.AddWarning("Missing sharding strategy in metadata");
        }
    }
}
