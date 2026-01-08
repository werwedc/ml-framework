namespace MLFramework.Communication.PointToPoint.Async;

using MLFramework.Communication.PointToPoint;
using RitterFramework.Core.Tensor;

/// <summary>
/// Asynchronous send operation
/// </summary>
public static class SendAsync
{
    /// <summary>
    /// Send tensor asynchronously to a specific rank
    /// </summary>
    public static ICommunicationHandle SendTensorAsync(
        IPointToPointCommunication backend,
        Tensor tensor,
        int destinationRank,
        int tag = 0)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (destinationRank < 0 || destinationRank >= backend.WorldSize)
            throw new ArgumentOutOfRangeException(nameof(destinationRank),
                $"Destination rank {destinationRank} is out of range [0, {backend.WorldSize})");

        if (destinationRank == backend.Rank)
            throw new ArgumentException("Cannot send to self");

        return backend.SendAsync(tensor, destinationRank, tag);
    }
}
