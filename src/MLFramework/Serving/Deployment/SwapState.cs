namespace MLFramework.Serving.Deployment;

/// <summary>
/// States for a swap operation
/// </summary>
public enum SwapState
{
    /// <summary>
    /// Swap has not started
    /// </summary>
    NotStarted,

    /// <summary>
    /// Loading the new version in the background
    /// </summary>
    LoadingNewVersion,

    /// <summary>
    /// Transitioning traffic from old to new version
    /// </summary>
    Transitioning,

    /// <summary>
    /// Waiting for old version to drain
    /// </summary>
    OldVersionDraining,

    /// <summary>
    /// Swap completed successfully
    /// </summary>
    Completed,

    /// <summary>
    /// Swap failed
    /// </summary>
    Failed
}
