namespace MLFramework.Fusion.Backends;

/// <summary>
/// Interface for fusion backend registry
/// </summary>
public interface IFusionBackendRegistry
{
    /// <summary>
    /// Registers a backend
    /// </summary>
    /// <param name="backend">Backend to register</param>
    void Register(IFusionBackend backend);

    /// <summary>
    /// Unregisters a backend
    /// </summary>
    /// <param name="backendName">Name of the backend to unregister</param>
    void Unregister(string backendName);

    /// <summary>
    /// Gets a backend by name
    /// </summary>
    /// <param name="backendName">Name of the backend</param>
    /// <returns>Backend if found, null otherwise</returns>
    IFusionBackend? GetBackend(string backendName);

    /// <summary>
    /// Gets the default backend
    /// </summary>
    /// <returns>Default backend</returns>
    IFusionBackend GetDefaultBackend();

    /// <summary>
    /// Sets the default backend
    /// </summary>
    /// <param name="backendName">Name of the backend to set as default</param>
    void SetDefaultBackend(string backendName);

    /// <summary>
    /// Gets all registered backends
    /// </summary>
    /// <returns>List of all registered backends</returns>
    IReadOnlyList<IFusionBackend> GetAllBackends();

    /// <summary>
    /// Finds a backend capable of fusing the operations
    /// </summary>
    /// <param name="operations">Operations to find a capable backend for</param>
    /// <returns>Capable backend if found, null otherwise</returns>
    IFusionBackend? FindCapableBackend(IReadOnlyList<Operation> operations);
}
