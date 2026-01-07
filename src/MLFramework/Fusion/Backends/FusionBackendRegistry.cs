namespace MLFramework.Fusion.Backends;

/// <summary>
/// Implementation of fusion backend registry
/// </summary>
public class FusionBackendRegistry : IFusionBackendRegistry
{
    private readonly Dictionary<string, IFusionBackend> _backends = new();
    private string _defaultBackendName = "Triton";
    private readonly object _lock = new();

    /// <summary>
    /// Registers a backend
    /// </summary>
    /// <param name="backend">Backend to register</param>
    public void Register(IFusionBackend backend)
    {
        lock (_lock)
        {
            _backends[backend.Name] = backend;
        }
    }

    /// <summary>
    /// Unregisters a backend
    /// </summary>
    /// <param name="backendName">Name of the backend to unregister</param>
    public void Unregister(string backendName)
    {
        lock (_lock)
        {
            _backends.Remove(backendName);
        }
    }

    /// <summary>
    /// Gets a backend by name
    /// </summary>
    /// <param name="backendName">Name of the backend</param>
    /// <returns>Backend if found, null otherwise</returns>
    public IFusionBackend? GetBackend(string backendName)
    {
        lock (_lock)
        {
            return _backends.TryGetValue(backendName, out var backend) ? backend : null;
        }
    }

    /// <summary>
    /// Gets the default backend
    /// </summary>
    /// <returns>Default backend</returns>
    public IFusionBackend GetDefaultBackend()
    {
        lock (_lock)
        {
            if (!_backends.TryGetValue(_defaultBackendName, out var backend))
            {
                throw new InvalidOperationException($"Default backend '{_defaultBackendName}' not registered");
            }
            return backend;
        }
    }

    /// <summary>
    /// Sets the default backend
    /// </summary>
    /// <param name="backendName">Name of the backend to set as default</param>
    public void SetDefaultBackend(string backendName)
    {
        lock (_lock)
        {
            if (!_backends.ContainsKey(backendName))
            {
                throw new InvalidOperationException($"Backend '{backendName}' is not registered");
            }
            _defaultBackendName = backendName;
        }
    }

    /// <summary>
    /// Gets all registered backends
    /// </summary>
    /// <returns>List of all registered backends</returns>
    public IReadOnlyList<IFusionBackend> GetAllBackends()
    {
        lock (_lock)
        {
            return _backends.Values.ToList();
        }
    }

    /// <summary>
    /// Finds a backend capable of fusing the operations
    /// </summary>
    /// <param name="operations">Operations to find a capable backend for</param>
    /// <returns>Capable backend if found, null otherwise</returns>
    public IFusionBackend? FindCapableBackend(IReadOnlyList<Operation> operations)
    {
        lock (_lock)
        {
            foreach (var backend in _backends.Values)
            {
                if (backend.CanFuse(operations))
                {
                    return backend;
                }
            }
        }

        return null;
    }
}
