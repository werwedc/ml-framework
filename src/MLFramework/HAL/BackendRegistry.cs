namespace MLFramework.HAL;

/// <summary>
/// Global registry for compute backends
/// </summary>
public static class BackendRegistry
{
    private static readonly Dictionary<DeviceType, IBackend> _backends = new();
    private static readonly object _registryLock = new();

    /// <summary>
    /// Register a backend for a specific device type
    /// </summary>
    /// <exception cref="ArgumentException">Thrown if backend already registered</exception>
    public static void Register(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);

        lock (_registryLock)
        {
            if (_backends.ContainsKey(backend.Type))
            {
                throw new ArgumentException(
                    $"Backend for device type {backend.Type} already registered");
            }

            _backends[backend.Type] = backend;

            // Initialize the backend
            if (backend.IsAvailable)
            {
                backend.Initialize();
            }
        }
    }

    /// <summary>
    /// Get the registered backend for a device type
    /// </summary>
    /// <returns>Backend instance, or null if not registered</returns>
    public static IBackend? GetBackend(DeviceType type)
    {
        lock (_registryLock)
        {
            return _backends.TryGetValue(type, out var backend) ? backend : null;
        }
    }

    /// <summary>
    /// Get all available device types (with registered backends)
    /// </summary>
    public static IEnumerable<DeviceType> GetAvailableDevices()
    {
        lock (_registryLock)
        {
            return _backends.Values
                .Where(b => b.IsAvailable)
                .Select(b => b.Type)
                .ToList();
        }
    }

    /// <summary>
    /// Check if a device type has a registered and available backend
    /// </summary>
    public static bool IsDeviceAvailable(DeviceType type)
    {
        return GetAvailableDevices().Contains(type);
    }

    /// <summary>
    /// Clear all registered backends (for testing)
    /// </summary>
    public static void Clear()
    {
        lock (_registryLock)
        {
            _backends.Clear();
        }
    }
}
