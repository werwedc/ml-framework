namespace MLFramework.Fusion.Backends;

/// <summary>
/// Factory for creating fusion backends
/// </summary>
public static class FusionBackendFactory
{
    private static readonly Dictionary<FusionBackendType, Func<IFusionBackend>> _creators = new();

    static FusionBackendFactory()
    {
        RegisterDefaults();
    }

    /// <summary>
    /// Registers a backend creator function
    /// </summary>
    /// <param name="type">Backend type</param>
    /// <param name="creator">Function to create the backend</param>
    public static void RegisterBackend(FusionBackendType type, Func<IFusionBackend> creator)
    {
        _creators[type] = creator;
    }

    /// <summary>
    /// Creates a backend of the specified type
    /// </summary>
    /// <param name="type">Backend type to create</param>
    /// <param name="config">Backend configuration</param>
    /// <returns>Created and initialized backend</returns>
    public static IFusionBackend CreateBackend(FusionBackendType type, BackendConfiguration config)
    {
        if (!_creators.TryGetValue(type, out var creator))
        {
            throw new ArgumentException($"Backend type {type} is not registered", nameof(type));
        }

        var backend = creator();
        backend.Initialize(config);
        return backend;
    }

    /// <summary>
    /// Registers default backend implementations
    /// </summary>
    public static void RegisterDefaults()
    {
        // Register Triton backend
        RegisterBackend(FusionBackendType.Triton, () =>
            new TritonBackend(
                new MockTritonCompiler(),
                new MockTritonAutotuner(),
                new ConsoleLogger()));

        // Register XLA backend
        RegisterBackend(FusionBackendType.XLA, () =>
            new XLABackend(
                new MockXLACompiler(),
                new ConsoleLogger()));
    }
}
