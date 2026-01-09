namespace MLFramework.Utilities;

using System.Collections.Generic;

/// <summary>
/// Diagnostic information about current deterministic state
/// </summary>
public class DiagnosticInfo
{
    /// <summary>
    /// Current deterministic mode flags
    /// </summary>
    public DeterministicModeFlags DeterministicMode { get; set; }

    /// <summary>
    /// Current global seed
    /// </summary>
    public int CurrentSeed { get; set; }

    /// <summary>
    /// Number of devices seeded
    /// </summary>
    public int DeviceCount { get; set; }

    /// <summary>
    /// Number of workers seeded
    /// </summary>
    public int WorkerCount { get; set; }

    /// <summary>
    /// Performance impact estimate
    /// </summary>
    public string PerformanceImpact { get; set; } = string.Empty;

    /// <summary>
    /// Known non-deterministic operations in use
    /// </summary>
    public List<string> NonDeterministicOperations { get; set; }

    public DiagnosticInfo()
    {
        NonDeterministicOperations = new List<string>();
    }
}
