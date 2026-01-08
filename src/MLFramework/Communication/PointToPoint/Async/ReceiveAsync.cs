namespace MLFramework.Communication.PointToPoint.Async;

using MLFramework.Communication.PointToPoint;
using RitterFramework.Core.Tensor;

/// <summary>
/// Asynchronous receive operation
/// </summary>
public static class ReceiveAsync
{
    /// <summary>
    /// Receive tensor asynchronously from a specific rank
    /// </summary>
    public static ICommunicationHandle ReceiveTensorAsync(
        IPointToPointCommunication backend,
        int sourceRank,
        int tag = 0)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (sourceRank < 0 || sourceRank >= backend.WorldSize)
            throw new ArgumentOutOfRangeException(nameof(sourceRank),
                $"Source rank {sourceRank} is out of range [0, {backend.WorldSize})");

        if (sourceRank == backend.Rank)
            throw new ArgumentException("Cannot receive from self");

        return backend.ReceiveAsync(sourceRank, tag);
    }

    /// <summary>
    /// Receive tensor asynchronously with known shape (more efficient)
    /// </summary>
    public static ICommunicationHandle ReceiveTensorAsync(
        IPointToPointCommunication backend,
        int sourceRank,
        Tensor template,
        int tag = 0)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (template == null)
            throw new ArgumentNullException(nameof(template));

        if (sourceRank < 0 || sourceRank >= backend.WorldSize)
            throw new ArgumentOutOfRangeException(nameof(sourceRank),
                $"Source rank {sourceRank} is out of range [0, {backend.WorldSize})");

        if (sourceRank == backend.Rank)
            throw new ArgumentException("Cannot receive from self");

        return backend.ReceiveAsync(sourceRank, template, tag);
    }
}
