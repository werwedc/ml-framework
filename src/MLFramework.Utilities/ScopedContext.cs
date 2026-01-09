namespace MLFramework.Utilities;

/// <summary>
/// Context manager for scoped determinism behavior
/// </summary>
public sealed class ScopedContext : IDisposable
{
    private readonly SeedManager _seedManager;
    private readonly int _previousSeed;
    private bool _disposed;

    /// <summary>
    /// Creates a new scoped context
    /// </summary>
    /// <param name="seedManager">The seed manager instance</param>
    /// <param name="previousSeed">The seed to restore on dispose</param>
    internal ScopedContext(SeedManager seedManager, int previousSeed)
    {
        _seedManager = seedManager;
        _previousSeed = previousSeed;
        _disposed = false;
    }

    /// <summary>
    /// Restores the previous state when exiting the scope
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;

        _seedManager.SetGlobalSeed(_previousSeed);

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
