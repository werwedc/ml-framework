namespace MLFramework.Core;

/// <summary>
/// Global configuration for diagnostic and error reporting behavior.
/// Controls enhanced error reporting, suggested fixes, shape tracking, and debug output.
/// </summary>
public class DiagnosticsConfiguration
{
    private static readonly Lazy<DiagnosticsConfiguration> _instance =
        new Lazy<DiagnosticsConfiguration>(() => new DiagnosticsConfiguration());

    /// <summary>
    /// Gets the singleton instance of the diagnostics configuration.
    /// </summary>
    public static DiagnosticsConfiguration Instance => _instance.Value;

    private DiagnosticsConfiguration()
    {
        // Private constructor for singleton pattern
    }

    /// <summary>
    /// Enables or disables enhanced error reporting with detailed shape information.
    /// When enabled, tensor operations will perform comprehensive shape validation
    /// and provide detailed error messages.
    /// </summary>
    public bool EnhancedErrorReporting { get; set; } = false;

    /// <summary>
    /// Enables or disables inclusion of suggested fixes in error reports.
    /// When enabled, shape validation errors will include actionable suggestions
    /// for resolving the issue.
    /// </summary>
    public bool IncludeSuggestions { get; set; } = true;

    /// <summary>
    /// Enables or disables shape tracking for tensor operations.
    /// When enabled, tensors will track their source operation and layer information
    /// for better error context. This may impact performance.
    /// </summary>
    public bool EnableShapeTracking { get; set; } = false;

    /// <summary>
    /// Gets or sets the maximum depth of context tracking.
    /// 0 = no context tracking, 1 = immediate parent only, higher values track deeper call stacks.
    /// Default is 1.
    /// </summary>
    public int ContextTrackingDepth { get; set; } = 1;

    /// <summary>
    /// Enables or disables debug mode with verbose output.
    /// When enabled, additional diagnostic information will be logged during operations.
    /// </summary>
    public bool DebugMode { get; set; } = false;

    /// <summary>
    /// Enables or disables logging of shape information.
    /// When enabled, shape changes and validation results will be logged.
    /// </summary>
    public bool LogShapeInformation { get; set; } = false;

    /// <summary>
    /// Resets the configuration to default values.
    /// </summary>
    public void ResetToDefaults()
    {
        EnhancedErrorReporting = false;
        IncludeSuggestions = true;
        EnableShapeTracking = false;
        ContextTrackingDepth = 1;
        DebugMode = false;
        LogShapeInformation = false;
    }
}
