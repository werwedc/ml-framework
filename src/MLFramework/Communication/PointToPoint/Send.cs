namespace MLFramework.Communication.PointToPoint;

using RitterFramework.Core.Tensor;

/// <summary>
/// Synchronous send operation
/// </summary>
public static class Send
{
    /// <summary>
    /// Send tensor to a specific rank
    /// </summary>
    public static void SendTensor(
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

        if (tag < 0)
            throw new ArgumentOutOfRangeException(nameof(tag), "Tag must be non-negative");

        backend.Send(tensor, destinationRank, tag);
    }

    /// <summary>
    /// Send multiple tensors
    /// </summary>
    public static void SendTensors(
        IPointToPointCommunication backend,
        IEnumerable<Tensor> tensors,
        int destinationRank,
        int tag = 0)
    {
        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        foreach (var tensor in tensors)
        {
            SendTensor(backend, tensor, destinationRank, tag++);
        }
    }
}
