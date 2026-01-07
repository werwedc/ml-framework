namespace MLFramework.Communication;

using RitterFramework.Core.Tensor;

/// <summary>
/// Represents a handle to an ongoing communication operation
/// </summary>
public interface ICommunicationHandle
{
    /// <summary>
    /// Returns true if the operation has completed
    /// </summary>
    bool IsCompleted { get; }

    /// <summary>
    /// Wait for the operation to complete
    /// </summary>
    void Wait();

    /// <summary>
    /// Wait for the operation to complete with timeout
    /// </summary>
    bool TryWait(int timeoutMs);

    /// <summary>
    /// Get the result tensor (only valid after completion)
    /// </summary>
    Tensor GetResult();
}
