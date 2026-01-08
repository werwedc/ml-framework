namespace MLFramework.Communication.PointToPoint;

using RitterFramework.Core.Tensor;

/// <summary>
/// Synchronous receive operation
/// </summary>
public static class Receive
{
    /// <summary>
    /// Receive tensor from a specific rank
    /// </summary>
    public static Tensor ReceiveTensor(
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

        if (tag < 0)
            throw new ArgumentOutOfRangeException(nameof(tag), "Tag must be non-negative");

        return backend.Receive(sourceRank, tag);
    }

    /// <summary>
    /// Receive tensor with known shape (more efficient)
    /// </summary>
    public static Tensor ReceiveTensor(
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

        if (tag < 0)
            throw new ArgumentOutOfRangeException(nameof(tag), "Tag must be non-negative");

        return backend.Receive(sourceRank, template, tag);
    }

    /// <summary>
    /// Receive multiple tensors
    /// </summary>
    public static List<Tensor> ReceiveTensors(
        IPointToPointCommunication backend,
        int sourceRank,
        int count,
        int tag = 0)
    {
        if (count < 0)
            throw new ArgumentOutOfRangeException(nameof(count), "Count must be non-negative");

        var result = new List<Tensor>();
        for (int i = 0; i < count; i++)
        {
            result.Add(ReceiveTensor(backend, sourceRank, tag + i));
        }
        return result;
    }
}
