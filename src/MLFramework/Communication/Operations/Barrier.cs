namespace MLFramework.Communication.Operations;

using MLFramework.Communication;
using MLFramework.Distributed.Communication;
using System;
using System.Threading.Tasks;

/// <summary>
/// Synchronization barrier operation
/// </summary>
public static class Barrier
{
    /// <summary>
    /// Block until all ranks in the process group reach this point
    /// </summary>
    /// <param name="backend">Communication backend</param>
    /// <param name="group">Process group (default: world)</param>
    public static void Synchronize(
        ICommunicationBackend backend,
        ProcessGroup? group = null)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        backend.Barrier();
    }

    /// <summary>
    /// Barrier with timeout
    /// </summary>
    /// <returns>True if barrier completed, false if timeout</returns>
    public static bool TrySynchronize(
        ICommunicationBackend backend,
        int timeoutMs,
        ProcessGroup? group = null)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        // This would need to be implemented by the backend
        // For now, we'll use a simple implementation
        var task = Task.Run(() => backend.Barrier());

        try
        {
            task.Wait(timeoutMs);
            return true;
        }
        catch (AggregateException ex) when (ex.InnerException is TimeoutException)
        {
            return false;
        }
        catch (TimeoutException)
        {
            return false;
        }
    }
}
