using MLFramework.Core;

namespace MLFramework;

/// <summary>
/// Global API for configuring and managing ML Framework diagnostics.
/// </summary>
public static class MLFramework
{
    private static bool _diagnosticsEnabled = false;

    /// <summary>
    /// Enables diagnostics with default settings.
    /// Default settings:
    /// - EnhancedErrorReporting: false
    /// - IncludeSuggestions: true
    /// - EnableShapeTracking: false
    /// - ContextTrackingDepth: 1
    /// - DebugMode: false
    /// - LogShapeInformation: false
    /// </summary>
    public static void EnableDiagnostics()
    {
        EnableDiagnostics(new DiagnosticsConfiguration());
    }

    /// <summary>
    /// Enables diagnostics with custom settings.
    /// </summary>
    /// <param name="config">The configuration to use for diagnostics.</param>
    public static void EnableDiagnostics(DiagnosticsConfiguration config)
    {
        if (config == null)
        {
            throw new ArgumentNullException(nameof(config));
        }

        // Copy custom settings to the singleton instance
        DiagnosticsConfiguration.Instance.EnhancedErrorReporting = config.EnhancedErrorReporting;
        DiagnosticsConfiguration.Instance.IncludeSuggestions = config.IncludeSuggestions;
        DiagnosticsConfiguration.Instance.EnableShapeTracking = config.EnableShapeTracking;
        DiagnosticsConfiguration.Instance.ContextTrackingDepth = config.ContextTrackingDepth;
        DiagnosticsConfiguration.Instance.DebugMode = config.DebugMode;
        DiagnosticsConfiguration.Instance.LogShapeInformation = config.LogShapeInformation;

        _diagnosticsEnabled = true;

        if (config.DebugMode)
        {
            System.Diagnostics.Debug.WriteLine("ML Framework Diagnostics enabled with custom configuration.");
        }
    }

    /// <summary>
    /// Disables diagnostics and resets configuration to defaults.
    /// </summary>
    public static void DisableDiagnostics()
    {
        _diagnosticsEnabled = false;
        DiagnosticsConfiguration.Instance.ResetToDefaults();

        System.Diagnostics.Debug.WriteLine("ML Framework Diagnostics disabled.");
    }

    /// <summary>
    /// Gets the current diagnostics configuration.
    /// </summary>
    /// <returns>The current DiagnosticsConfiguration instance.</returns>
    public static DiagnosticsConfiguration GetDiagnosticsConfiguration()
    {
        return DiagnosticsConfiguration.Instance;
    }

    /// <summary>
    /// Checks whether diagnostics are currently enabled.
    /// </summary>
    /// <returns>True if diagnostics are enabled, false otherwise.</returns>
    public static bool IsDiagnosticsEnabled()
    {
        return _diagnosticsEnabled;
    }

    /// <summary>
    /// Enables enhanced error reporting specifically.
    /// </summary>
    public static void EnableEnhancedErrorReporting()
    {
        EnableDiagnostics();
        DiagnosticsConfiguration.Instance.EnhancedErrorReporting = true;
    }

    /// <summary>
    /// Enables shape tracking specifically.
    /// </summary>
    public static void EnableShapeTracking()
    {
        EnableDiagnostics();
        DiagnosticsConfiguration.Instance.EnableShapeTracking = true;
    }

    /// <summary>
    /// Enables debug mode specifically.
    /// </summary>
    public static void EnableDebugMode()
    {
        EnableDiagnostics();
        DiagnosticsConfiguration.Instance.DebugMode = true;
        DiagnosticsConfiguration.Instance.LogShapeInformation = true;
    }
}
