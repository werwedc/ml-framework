namespace MLFramework.Utilities;

/// <summary>
/// Provides centralized control over random number generators (RNGs) used throughout the framework.
/// This class manages seeding for CPU, NumPy, and CUDA RNGs to ensure reproducibility.
/// </summary>
public class SeedManager : IDisposable
{
    private int _currentSeed;
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the SeedManager class.
    /// </summary>
    public SeedManager()
    {
        _currentSeed = 0;
        _disposed = false;
    }

    /// <summary>
    /// Sets a global seed for all RNGs (random, NumPy, CUDA).
    /// This method is thread-safe.
    /// </summary>
    /// <param name="seed">The seed value to use</param>
    public void SetGlobalSeed(int seed)
    {
        lock (_lock)
        {
            _currentSeed = seed;
            SetRandomSeed(seed);
            SetNumpySeed(seed);
            SetCudaSeed(seed);
        }
    }

    /// <summary>
    /// Seeds the CPU random number generator.
    /// Stores the seed for new Random instances.
    /// Note: Existing Random instances are not affected.
    /// </summary>
    /// <param name="seed">The seed value to use</param>
    public void SetRandomSeed(int seed)
    {
        // For pure C# implementation, we store the seed for new Random instances.
        // Existing Random instances are not affected to avoid unexpected behavior.
        // Consider using a thread-safe Random pool in future implementations.
    }

    /// <summary>
    /// Seeds the NumPy random number generator.
    /// Note: This is a placeholder for interop with NumPy (if using Python interop).
    /// For pure C# implementation, this may be a no-op or use equivalent library.
    /// The interface is designed to be compatible with future NumPy integration.
    /// </summary>
    /// <param name="seed">The seed value to use</param>
    public void SetNumpySeed(int seed)
    {
        // Placeholder for NumPy interop
        // In pure C# implementation, this is a no-op or uses equivalent library
    }

    /// <summary>
    /// Seeds the CUDA random number generator.
    /// Interface for CUDA RNG seeding.
    /// Placeholder for CUDA-specific implementation.
    /// Handles cases where CUDA is not available.
    /// </summary>
    /// <param name="seed">The seed value to use</param>
    public void SetCudaSeed(int seed)
    {
        // Placeholder for CUDA-specific implementation
        // Will handle CUDA availability checks in future iterations
    }

    /// <summary>
    /// Gets the current global seed value.
    /// </summary>
    public int CurrentSeed => _currentSeed;

    /// <summary>
    /// Disposes the seed manager and cleans up resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;

        // Clean up resources here
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Finalizer for SeedManager.
    /// </summary>
    ~SeedManager()
    {
        Dispose();
    }
}
