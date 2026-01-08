namespace MLFramework.Communication.PointToPoint;

using RitterFramework.Core.Tensor;

/// <summary>
/// Common send-receive patterns
/// </summary>
public static class SendReceivePair
{
    /// <summary>
    /// Send to one rank and receive from another (non-blocking)
    /// </summary>
    public static (ICommunicationHandle sendHandle, ICommunicationHandle receiveHandle) SendReceiveAsync(
        IPointToPointCommunication backend,
        Tensor sendTensor,
        int destinationRank,
        int sourceRank,
        int sendTag = 0,
        int receiveTag = 0)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (sendTensor == null)
            throw new ArgumentNullException(nameof(sendTensor));

        if (destinationRank == sourceRank)
            throw new ArgumentException("Destination and source ranks must be different");

        // Post receive first to avoid deadlock
        var receiveHandle = backend.ReceiveAsync(sourceRank, receiveTag);
        var sendHandle = backend.SendAsync(sendTensor, destinationRank, sendTag);

        return (sendHandle, receiveHandle);
    }

    /// <summary>
    /// Send to one rank and receive from another (blocking)
    /// </summary>
    public static Tensor SendReceive(
        IPointToPointCommunication backend,
        Tensor sendTensor,
        int destinationRank,
        int sourceRank,
        int sendTag = 0,
        int receiveTag = 0)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (sendTensor == null)
            throw new ArgumentNullException(nameof(sendTensor));

        if (destinationRank == sourceRank)
            throw new ArgumentException("Destination and source ranks must be different");

        // Use async to avoid deadlock
        var (sendHandle, receiveHandle) = SendReceiveAsync(
            backend, sendTensor, destinationRank, sourceRank, sendTag, receiveTag);

        // Wait for both operations
        sendHandle.Wait();
        receiveHandle.Wait();

        return receiveHandle.GetResult();
    }

    /// <summary>
    /// Ring send-receive pattern (each rank sends to next, receives from previous)
    /// </summary>
    public static Tensor RingSendReceive(
        IPointToPointCommunication backend,
        Tensor sendTensor,
        int tag = 0)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (sendTensor == null)
            throw new ArgumentNullException(nameof(sendTensor));

        int nextRank = (backend.Rank + 1) % backend.WorldSize;
        int prevRank = (backend.Rank - 1 + backend.WorldSize) % backend.WorldSize;

        return SendReceive(backend, sendTensor, nextRank, prevRank, tag, tag);
    }
}
