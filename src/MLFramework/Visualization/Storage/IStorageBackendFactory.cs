namespace MachineLearning.Visualization.Storage;

/// <summary>
/// Interface for creating storage backend instances
/// </summary>
public interface IStorageBackendFactory
{
    /// <summary>
    /// Creates a storage backend of the specified type
    /// </summary>
    /// <param name="backendType">Type of backend to create (e.g., "file", "memory", "remote")</param>
    /// <param name="connectionString">Connection string for the backend</param>
    /// <returns>A new storage backend instance</returns>
    IStorageBackend CreateBackend(string backendType, string connectionString);
}
