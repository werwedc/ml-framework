namespace MLFramework.Fusion;

/// <summary>
/// Interface for fusion statistics collection
/// </summary>
public interface IFusionStatistics
{
    /// <summary>
    /// Gets statistics for the current session
    /// </summary>
    FusionStatistics GetCurrentStatistics();

    /// <summary>
    /// Resets statistics
    /// </summary>
    void Reset();

    /// <summary>
    /// Logs fusion decisions for debugging
    /// </summary>
    void LogFusionDecisions();
}
