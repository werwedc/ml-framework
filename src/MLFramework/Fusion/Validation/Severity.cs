namespace MLFramework.Fusion.Validation;

/// <summary>
/// Severity level of constraint violations
/// </summary>
public enum Severity
{
    /// <summary>Error - Cannot fuse operations</summary>
    Error,

    /// <summary>Warning - Can fuse but with caveats</summary>
    Warning
}
