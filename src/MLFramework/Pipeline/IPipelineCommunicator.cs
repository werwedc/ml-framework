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
        /// Gets the rank of this stage (0 to WorldSize-1)
        /// </summary>
        int Rank { get; }

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
    }
}
