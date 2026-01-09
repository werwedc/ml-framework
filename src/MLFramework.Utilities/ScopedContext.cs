namespace MLFramework.Utilities;

/// <summary>
/// Context manager for scoped determinism behavior
/// </summary>
public sealed class ScopedContext : IDisposable
{
    private readonly SeedManager _seedManager;
    private readonly DeterministicModeFlags _previousMode;
    private readonly RNGSnapshot? _previousRngState;
    private readonly int? _previousSeed;
    private readonly bool _restoreState;
    private bool _disposed;

    /// <summary>
    /// Creates a new scoped context for global seed only.
    /// </summary>
    /// <param name="seedManager">The seed manager instance</param>
    /// <param name="previousSeed">The seed to restore on dispose</param>
    internal ScopedContext(SeedManager seedManager, int previousSeed)
    {
        _seedManager = seedManager;
        _previousMode = DeterministicModeFlags.None;
        _previousRngState = null;
        _previousSeed = previousSeed;
        _restoreState = false;
        _disposed = false;
    }

    /// <summary>
    /// Creates a new scoped context with deterministic mode support.
    /// </summary>
    /// <param name="seedManager">The seed manager instance</param>
    /// <param name="previousMode">The mode to restore on dispose</param>
    /// <param name="previousRngState">The RNG state to restore (optional)</param>
    /// <param name="restoreState">Whether to restore RNG state on exit</param>
    internal ScopedContext(
        SeedManager seedManager,
        DeterministicModeFlags previousMode,
        RNGSnapshot? previousRngState = null,
        bool restoreState = false)
    {
        _seedManager = seedManager;
        _previousMode = previousMode;
        _previousRngState = previousRngState;
        _previousSeed = null;
        _restoreState = restoreState;
        _disposed = false;
    }

    /// <summary>
    /// Restores the previous state when exiting the scope.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;

        // Restore deterministic mode if it was set
        if (_previousMode != DeterministicModeFlags.None || _previousSeed == null)
        {
            _seedManager.SetDeterministicMode(_previousMode);
        }

        // Restore RNG state if requested
        if (_restoreState && _previousRngState != null)
        {
            _seedManager.RestoreRNGState(_previousRngState);
        }
        // Otherwise restore seed if it was set
        else if (_previousSeed.HasValue)
        {
            _seedManager.SetGlobalSeed(_previousSeed.Value);
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
