using RitterFramework.Core.Tensor;
using System;
using System.Threading.Tasks;

namespace MLFramework.Distributed
{
    /// <summary>
    /// Interface for process groups used in distributed training.
    /// </summary>
    public interface IProcessGroup : IDisposable
    {
        /// <summary>
        /// Gets the rank of this process in the process group.
        /// </summary>
        int Rank { get; }

        /// <summary>
        /// Gets the total number of processes in the process group.
        /// </summary>
        int WorldSize { get; }

        /// <summary>
        /// Gets the communication backend used by this process group.
        /// </summary>
        ICommunicationBackend Backend { get; }

        /// <summary>
        /// Performs an AllReduce operation on the given tensor.
        /// </summary>
        void AllReduce(Tensor tensor, ReduceOp op = ReduceOp.Sum);

        /// <summary>
        /// Performs a Broadcast operation, sending data from root to all processes.
        /// </summary>
        void Broadcast(Tensor tensor, int root = 0);

        /// <summary>
        /// Synchronizes all processes in the process group.
        /// </summary>
        void Barrier();

        /// <summary>
        /// Performs an asynchronous AllReduce operation.
        /// </summary>
        Task AllReduceAsync(Tensor tensor, ReduceOp op = ReduceOp.Sum);

        /// <summary>
        /// Performs an asynchronous Broadcast operation.
        /// </summary>
        Task BroadcastAsync(Tensor tensor, int root = 0);

        /// <summary>
        /// Performs an asynchronous barrier.
        /// </summary>
        Task BarrierAsync();

        /// <summary>
        /// Sends a tensor to the specified destination rank.
        /// </summary>
        void Send(Tensor tensor, int dst);

        /// <summary>
        /// Receives a tensor from the specified source rank.
        /// </summary>
        void Recv(Tensor tensor, int src);

        /// <summary>
        /// Asynchronously sends a tensor to the specified destination rank.
        /// </summary>
        Task SendAsync(Tensor tensor, int dst);

        /// <summary>
        /// Asynchronously receives a tensor from the specified source rank.
        /// </summary>
        Task RecvAsync(Tensor tensor, int src);

        /// <summary>
        /// Destroys the process group and releases resources.
        /// </summary>
        void Destroy();
    }

    /// <summary>
    /// Interface for communication backends.
    /// </summary>
    public interface ICommunicationBackend
    {
        /// <summary>
        /// Gets the name of the backend.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets whether the backend is available on this system.
        /// </summary>
        bool IsAvailable { get; }

        /// <summary>
        /// Gets the number of devices available for communication.
        /// </summary>
        int DeviceCount { get; }

        /// <summary>
        /// Gets whether the backend supports asynchronous operations.
        /// </summary>
        bool SupportsAsync { get; }

        /// <summary>
        /// Gets whether the backend supports GPU direct communication.
        /// </summary>
        bool SupportsGPUDirect { get; }

        /// <summary>
        /// Gets the buffer size limit for communication operations.
        /// </summary>
        long GetBufferSizeLimit();
    }

    /// <summary>
    /// Reduction operations for distributed communication.
    /// </summary>
    public enum ReduceOp
    {
        /// <summary>
        /// Sum of all values.
        /// </summary>
        Sum,

        /// <summary>
        /// Average of all values.
        /// </summary>
        Avg,

        /// <summary>
        /// Maximum of all values.
        /// </summary>
        Max,

        /// <summary>
        /// Minimum of all values.
        /// </summary>
        Min,

        /// <summary>
        /// Product of all values.
        /// </summary>
        Product
    }

    /// <summary>
    /// Exception thrown when a communication operation fails.
    /// </summary>
    public class CommunicationException : Exception
    {
        /// <summary>
        /// Gets the rank of the process that threw the exception.
        /// </summary>
        public int Rank { get; }

        /// <summary>
        /// Gets the name of the backend being used.
        /// </summary>
        public string BackendName { get; }

        public CommunicationException(string message, int rank, string backendName)
            : base(message)
        {
            Rank = rank;
            BackendName = backendName;
        }

        public CommunicationException(string message, int rank, string backendName, Exception innerException)
            : base(message, innerException)
        {
            Rank = rank;
            BackendName = backendName;
        }
    }
}
