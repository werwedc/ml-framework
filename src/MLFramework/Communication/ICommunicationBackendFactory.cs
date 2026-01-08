namespace MLFramework.Communication;

using MLFramework.Distributed.Communication;

/// <summary>
/// Interface for creating communication backends
/// </summary>
public interface ICommunicationBackendFactory
{
    /// <summary>
    /// Detect if this backend is available on current system
    /// </summary>
    bool IsAvailable();

    /// <summary>
    /// Create a backend instance with given configuration
    /// </summary>
    ICommunicationBackend Create(MLFramework.Distributed.Communication.CommunicationConfig config);

    /// <summary>
    /// Get the priority of this backend (higher = preferred)
    /// </summary>
    int Priority { get; }
}
