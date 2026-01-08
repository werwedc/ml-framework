namespace MLFramework.Communication.PointToPoint;

using MLFramework.Communication;
using RitterFramework.Core.Tensor;

/// <summary>
/// Interface for point-to-point communication operations
/// </summary>
public interface IPointToPointCommunication : ICommunicationBackend
{
    /// <summary>
    /// Send tensor to a specific rank
    /// </summary>
    /// <param name="tensor">Tensor to send</param>
    /// <param name="destinationRank">Rank to send to</param>
    /// <param name="tag">Message tag for matching (optional)</param>
    void Send(Tensor tensor, int destinationRank, int tag = 0);

    /// <summary>
    /// Receive tensor from a specific rank
    /// </summary>
    /// <param name="sourceRank">Rank to receive from</param>
    /// <param name="tag">Message tag for matching (optional)</param>
    /// <returns>Received tensor</returns>
    Tensor Receive(int sourceRank, int tag = 0);

    /// <summary>
    /// Receive tensor with known shape (more efficient)
    /// </summary>
    Tensor Receive(int sourceRank, Tensor template, int tag = 0);

    /// <summary>
    /// Non-blocking send
    /// </summary>
    ICommunicationHandle SendAsync(Tensor tensor, int destinationRank, int tag = 0);

    /// <summary>
    /// Non-blocking receive
    /// </summary>
    ICommunicationHandle ReceiveAsync(int sourceRank, int tag = 0);

    /// <summary>
    /// Non-blocking receive with known shape
    /// </summary>
    ICommunicationHandle ReceiveAsync(int sourceRank, Tensor template, int tag = 0);

    /// <summary>
    /// Probe for incoming message
    /// </summary>
    /// <param name="sourceRank">Rank to probe</param>
    /// <param name="tag">Message tag (use -1 for any tag)</param>
    /// <returns>Message info or null if no message</returns>
    MessageInfo? Probe(int sourceRank, int tag = 0);
}
