using System.Collections.Concurrent;

namespace MachineLearning.Visualization.Storage;

/// <summary>
/// Factory for creating storage backend instances
/// </summary>
public class StorageBackendFactory : IStorageBackendFactory
{
    private readonly ConcurrentDictionary<string, Type> _backendTypes;
    private readonly ConcurrentDictionary<string, Func<IStorageBackend>> _backendFactories;

    /// <summary>
    /// Initializes a new instance of StorageBackendFactory
    /// </summary>
    public StorageBackendFactory()
    {
        _backendTypes = new ConcurrentDictionary<string, Type>(StringComparer.OrdinalIgnoreCase);
        _backendFactories = new ConcurrentDictionary<string, Func<IStorageBackend>>(StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Registers a storage backend type
    /// </summary>
    /// <typeparam name="TBackend">Type of the backend (must implement IStorageBackend and have a parameterless constructor)</typeparam>
    /// <param name="backendType">String identifier for the backend type</param>
    public void RegisterBackend<TBackend>(string backendType) where TBackend : IStorageBackend, new()
    {
        if (string.IsNullOrWhiteSpace(backendType))
        {
            throw new ArgumentException("Backend type cannot be null or empty", nameof(backendType));
        }

        _backendTypes[backendType] = typeof(TBackend);
    }

    /// <summary>
    /// Registers a storage backend factory function
    /// </summary>
    /// <param name="backendType">String identifier for the backend type</param>
    /// <param name="factory">Factory function to create the backend</param>
    public void RegisterBackendFactory(string backendType, Func<IStorageBackend> factory)
    {
        if (string.IsNullOrWhiteSpace(backendType))
        {
            throw new ArgumentException("Backend type cannot be null or empty", nameof(backendType));
        }

        if (factory == null)
        {
            throw new ArgumentNullException(nameof(factory));
        }

        _backendFactories[backendType] = factory;
    }

    /// <summary>
    /// Creates a storage backend of the specified type
    /// </summary>
    /// <param name="backendType">Type of backend to create (e.g., "file", "memory", "remote")</param>
    /// <param name="connectionString">Connection string for the backend</param>
    /// <returns>A new storage backend instance</returns>
    public IStorageBackend CreateBackend(string backendType, string connectionString)
    {
        if (string.IsNullOrWhiteSpace(backendType))
        {
            throw new ArgumentException("Backend type cannot be null or empty", nameof(backendType));
        }

        // Try to get from factory
        if (_backendFactories.TryGetValue(backendType, out var factory))
        {
            var backend = factory();
            backend.Initialize(connectionString);
            return backend;
        }

        // Try to get from registered types
        if (_backendTypes.TryGetValue(backendType, out var backendTypeClass))
        {
            if (!typeof(IStorageBackend).IsAssignableFrom(backendTypeClass))
            {
                throw new InvalidOperationException(
                    $"Type {backendTypeClass.FullName} does not implement IStorageBackend");
            }

            var constructor = backendTypeClass.GetConstructor(Type.EmptyTypes);
            if (constructor == null)
            {
                throw new InvalidOperationException(
                    $"Type {backendTypeClass.FullName} does not have a parameterless constructor");
            }

            var backend = (IStorageBackend)constructor.Invoke(null);
            backend.Initialize(connectionString);
            return backend;
        }

        throw new InvalidOperationException(
            $"Unknown backend type: {backendType}. Available types: {string.Join(", ", GetAvailableBackendTypes())}");
    }

    /// <summary>
    /// Gets the list of available backend types
    /// </summary>
    /// <returns>List of registered backend type names</returns>
    public IEnumerable<string> GetAvailableBackendTypes()
    {
        return _backendTypes.Keys.Concat(_backendFactories.Keys).Distinct(StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Checks if a backend type is registered
    /// </summary>
    /// <param name="backendType">Backend type to check</param>
    /// <returns>True if the backend type is registered, false otherwise</returns>
    public bool IsBackendRegistered(string backendType)
    {
        if (string.IsNullOrWhiteSpace(backendType))
        {
            return false;
        }

        return _backendTypes.ContainsKey(backendType) || _backendFactories.ContainsKey(backendType);
    }
}
