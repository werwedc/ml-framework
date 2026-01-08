namespace MachineLearning.Checkpointing;

/// <summary>
/// Checker for version compatibility validation
/// </summary>
public class VersionCompatibilityChecker : ICompatibilityChecker
{
    private readonly string _supportedVersionRange;

    /// <summary>
    /// Create a new VersionCompatibilityChecker
    /// </summary>
    public VersionCompatibilityChecker(string supportedVersionRange = "1.0.x")
    {
        _supportedVersionRange = supportedVersionRange;
    }

    /// <summary>
    /// Name of the compatibility checker
    /// </summary>
    public string Name => "Version";

    /// <summary>
    /// Check if a checkpoint is compatible with the current version
    /// </summary>
    public async Task<ValidationResult> CheckCompatibilityAsync(
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default)
    {
        var result = new ValidationResult();

        try
        {
            var version = new Version(metadata.Version);
            var supportedVersion = new Version(_supportedVersionRange.Replace("x", "0"));

            if (version.Major != supportedVersion.Major)
            {
                result.AddError($"Incompatible checkpoint version: {metadata.Version} (supported: {_supportedVersionRange})");
            }
        }
        catch (Exception ex)
        {
            result.AddError($"Invalid version format: {ex.Message}");
        }

        return await Task.FromResult(result);
    }
}
