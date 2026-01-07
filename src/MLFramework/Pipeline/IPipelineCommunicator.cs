using System;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// Interface for pipeline communication between stages
    /// </summary>
    public interface IPipelineCommunicator : IDisposable
    {
        /// <summary>
        /// Gets the world size (total number of stages/devices)
        /// </summary>
        int WorldSize { get; }

        /// <summary>
        /// Rank of the current process/device
        /// </summary>
        int CurrentRank { get; }

        /// <summary>
        /// Send a tensor to a specific stage
        /// </summary>
        /// <param name="tensor">Tensor to send</param>
        /// <param name="destRank">Destination stage rank</param>
        Task SendAsync(Tensor tensor, int destRank);

        /// <summary>
        /// Receive a tensor from a specific stage
        /// </summary>
        /// <param name="srcRank">Source stage rank</param>
        Task<Tensor> ReceiveAsync(int srcRank);

        /// <summary>
        /// Broadcast a tensor from root to all stages
        /// </summary>
        /// <param name="tensor">Tensor to broadcast</param>
        /// <param name="root">Root stage rank</param>
        Task<Tensor> BroadcastAsync(Tensor tensor, int root);

        /// <summary>
        /// Synchronize all stages (barrier)
        /// </summary>
        Task BarrierAsync();

        /// <summary>
        /// Send forward activation to next stage asynchronously
        /// </summary>
        /// <param name="tensor">Tensor to send</param>
        /// <param name="destinationRank">Destination stage rank</param>
        Task<Tensor> SendForwardAsync(Tensor tensor, int destinationRank);

        /// <summary>
        /// Receive forward activation from previous stage asynchronously
        /// </summary>
        /// <param name="sourceRank">Source stage rank</param>
        Task<Tensor> ReceiveForwardAsync(int sourceRank);

        /// <summary>
        /// Send backward gradient to previous stage asynchronously
        /// </summary>
        /// <param name="tensor">Tensor to send</param>
        /// <param name="destinationRank">Destination stage rank</param>
        Task<Tensor> SendBackwardAsync(Tensor tensor, int destinationRank);

        /// <summary>
        /// Receive backward gradient from next stage asynchronously
        /// </summary>
        /// <param name="sourceRank">Source stage rank</param>
        Task<Tensor> ReceiveBackwardAsync(int sourceRank);
    }
}
