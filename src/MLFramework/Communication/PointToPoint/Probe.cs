namespace MLFramework.Communication.PointToPoint;

using System.Diagnostics;

/// <summary>
/// Probe for incoming messages
/// </summary>
public static class Probe
{
    /// <summary>
    /// Check if there's an incoming message from a specific rank
    /// </summary>
    /// <returns>Message info or null if no message available</returns>
    public static MessageInfo? ProbeMessage(
        IPointToPointCommunication backend,
        int sourceRank,
        int tag = 0)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (sourceRank < 0 || sourceRank >= backend.WorldSize)
            throw new ArgumentOutOfRangeException(nameof(sourceRank),
                $"Source rank {sourceRank} is out of range [0, {backend.WorldSize})");

        return backend.Probe(sourceRank, tag);
    }

    /// <summary>
    /// Wait for message with timeout
    /// </summary>
    /// <returns>Message info or null if timeout</returns>
    public static MessageInfo? WaitForMessage(
        IPointToPointCommunication backend,
        int sourceRank,
        int timeoutMs = 5000,
        int tag = 0)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        var stopwatch = Stopwatch.StartNew();
        while (stopwatch.ElapsedMilliseconds < timeoutMs)
        {
            var info = ProbeMessage(backend, sourceRank, tag);
            if (info != null)
            {
                return info;
            }
            Thread.Sleep(10);
        }
        return null;
    }
}
