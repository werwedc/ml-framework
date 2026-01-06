using MLFramework.Distributed.Gloo;
using MLFramework.Distributed.NCCL;
using RitterFramework.Core.Tensor;
using System;
using System.Threading.Tasks;

namespace MLFramework.Distributed
{
    /// <summary>
    /// Main process group management system for distributed training.
    /// Provides a singleton pattern for process group initialization and management.
    /// </summary>
    public class ProcessGroup : IProcessGroup, IDisposable
    {
        private static ProcessGroup? _defaultProcessGroup;
        private static readonly object _lock = new object();
        private readonly IProcessGroup _processGroupImpl;
        private readonly ICommunicationBackend _backend;
        private readonly int _rank;
        private readonly int _worldSize;
        private bool _disposed;

        // Private constructor - use static factory methods
        private ProcessGroup(IProcessGroup processGroupImpl, ICommunicationBackend backend, int rank, int worldSize)
        {
            _processGroupImpl = processGroupImpl ?? throw new ArgumentNullException(nameof(processGroupImpl));
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            _rank = rank;
            _worldSize = worldSize;
            _disposed = false;
        }

        /// <summary>
        /// Gets the rank of this process in the process group.
        /// </summary>
        public int Rank => _rank;

        /// <summary>
        /// Gets the total number of processes in the process group.
        /// </summary>
        public int WorldSize => _worldSize;

        /// <summary>
        /// Gets the communication backend used by this process group.
        /// </summary>
        public ICommunicationBackend Backend => _backend;

        /// <summary>
        /// Gets the default process group instance.
        /// Returns null if no process group has been initialized.
        /// </summary>
        public static ProcessGroup? Default => _defaultProcessGroup;

        /// <summary>
        /// Initialize a process group with the specified backend type.
        /// Reads configuration from environment variables.
        /// </summary>
        /// <param name="backendType">The type of backend to use (NCCL, Gloo, etc.).</param>
        /// <param name="initMethod">The initialization method to use ("env" for environment variables).</param>
        /// <returns>The initialized process group.</returns>
        /// <exception cref="InvalidOperationException">Thrown when a process group is already initialized.</exception>
        /// <exception cref="CommunicationException">Thrown when initialization fails.</exception>
        /// <exception cref="ArgumentException">Thrown when initialization parameters are invalid.</exception>
        public static ProcessGroup Init(BackendType backendType, string initMethod = "env")
        {
            lock (_lock)
            {
                if (_defaultProcessGroup != null)
                {
                    throw new InvalidOperationException(
                        "Process group is already initialized. Call ProcessGroup.Destroy() before creating a new one.");
                }

                if (initMethod != "env")
                {
                    throw new ArgumentException(
                        $"Initialization method '{initMethod}' is not supported. Only 'env' is supported.",
                        nameof(initMethod));
                }

                // Check backend availability
                if (!BackendFactory.IsBackendAvailable(backendType))
                {
                    throw new CommunicationException(
                        $"Backend {backendType} is not available on this system.",
                        GetEnvVar("RANK", 0),
                        backendType.ToString());
                }

                // Create backend
                var backend = BackendFactory.CreateBackend(backendType);

                // Read rank and world size from environment
                var rank = GetEnvVar("RANK", 0);
                var worldSize = GetEnvVar("WORLD_SIZE", 1);

                // Validate rank and world size
                if (worldSize <= 0)
                {
                    throw new ArgumentException(
                        "WORLD_SIZE must be positive. Check your environment variables.",
                        nameof(worldSize));
                }

                if (rank < 0 || rank >= worldSize)
                {
                    throw new ArgumentException(
                        $"RANK must be in range [0, {worldSize - 1}]. Check your environment variables.",
                        nameof(rank));
                }

                // Create the appropriate process group implementation
                IProcessGroup processGroupImpl = backendType switch
                {
                    BackendType.NCCL => new NCCLProcessGroup((NCCLBackend)backend),
                    BackendType.Gloo => new GlooProcessGroup((GlooBackend)backend),
                    _ => throw new ArgumentException($"Unsupported backend type: {backendType}")
                };

                _defaultProcessGroup = new ProcessGroup(processGroupImpl, backend, rank, worldSize);

                return _defaultProcessGroup;
            }
        }

        /// <summary>
        /// Destroy the current process group and release resources.
        /// Must be called before creating a new process group.
        /// </summary>
        public static void Destroy()
        {
            lock (_lock)
            {
                if (_defaultProcessGroup != null)
                {
                    _defaultProcessGroup.Dispose();
                    _defaultProcessGroup = null;
                }
            }
        }

        /// <summary>
        /// Performs an AllReduce operation on the given tensor.
        /// Modifies the tensor in-place with the reduced result.
        /// </summary>
        /// <param name="tensor">The tensor to reduce.</param>
        /// <param name="op">The reduction operation to perform (default: Sum).</param>
        public void AllReduce(Tensor tensor, ReduceOp op = ReduceOp.Sum)
        {
            EnsureNotDisposed();
            _processGroupImpl.AllReduce(tensor, op);
        }

        /// <summary>
        /// Performs a Broadcast operation, sending data from root to all processes.
        /// </summary>
        /// <param name="tensor">The tensor to broadcast.</param>
        /// <param name="root">The rank of the root process (default: 0).</param>
        public void Broadcast(Tensor tensor, int root = 0)
        {
            EnsureNotDisposed();
            _processGroupImpl.Broadcast(tensor, root);
        }

        /// <summary>
        /// Synchronizes all processes in the process group.
        /// </summary>
        public void Barrier()
        {
            EnsureNotDisposed();
            _processGroupImpl.Barrier();
        }

        /// <summary>
        /// Performs an asynchronous AllReduce operation.
        /// </summary>
        /// <param name="tensor">The tensor to reduce.</param>
        /// <param name="op">The reduction operation to perform (default: Sum).</param>
        /// <returns>A task that completes when the operation is done.</returns>
        public Task AllReduceAsync(Tensor tensor, ReduceOp op = ReduceOp.Sum)
        {
            EnsureNotDisposed();
            return _processGroupImpl.AllReduceAsync(tensor, op);
        }

        /// <summary>
        /// Performs an asynchronous Broadcast operation.
        /// </summary>
        /// <param name="tensor">The tensor to broadcast.</param>
        /// <param name="root">The rank of the root process (default: 0).</param>
        /// <returns>A task that completes when the operation is done.</returns>
        public Task BroadcastAsync(Tensor tensor, int root = 0)
        {
            EnsureNotDisposed();
            return _processGroupImpl.BroadcastAsync(tensor, root);
        }

        /// <summary>
        /// Performs an asynchronous barrier.
        /// </summary>
        /// <returns>A task that completes when all processes have reached the barrier.</returns>
        public Task BarrierAsync()
        {
            EnsureNotDisposed();
            return _processGroupImpl.BarrierAsync();
        }

        /// <summary>
        /// Sends a tensor to the specified destination rank.
        /// </summary>
        /// <param name="tensor">The tensor to send.</param>
        /// <param name="dst">The destination rank.</param>
        public void Send(Tensor tensor, int dst)
        {
            EnsureNotDisposed();
            _processGroupImpl.Send(tensor, dst);
        }

        /// <summary>
        /// Receives a tensor from the specified source rank.
        /// </summary>
        /// <param name="tensor">The tensor to receive into.</param>
        /// <param name="src">The source rank.</param>
        public void Recv(Tensor tensor, int src)
        {
            EnsureNotDisposed();
            _processGroupImpl.Recv(tensor, src);
        }

        /// <summary>
        /// Asynchronously sends a tensor to the specified destination rank.
        /// </summary>
        /// <param name="tensor">The tensor to send.</param>
        /// <param name="dst">The destination rank.</param>
        /// <returns>A task that completes when the send is done.</returns>
        public Task SendAsync(Tensor tensor, int dst)
        {
            EnsureNotDisposed();
            return _processGroupImpl.SendAsync(tensor, dst);
        }

        /// <summary>
        /// Asynchronously receives a tensor from the specified source rank.
        /// </summary>
        /// <param name="tensor">The tensor to receive into.</param>
        /// <param name="src">The source rank.</param>
        /// <returns>A task that completes when the receive is done.</returns>
        public Task RecvAsync(Tensor tensor, int src)
        {
            EnsureNotDisposed();
            return _processGroupImpl.RecvAsync(tensor, src);
        }

        /// <summary>
        /// Destroys the process group and releases resources.
        /// This is an explicit interface implementation to avoid conflict with the static Destroy() method.
        /// </summary>
        void IProcessGroup.Destroy()
        {
            lock (_lock)
            {
                if (!_disposed)
                {
                    _processGroupImpl.Destroy();
                    _disposed = true;
                }
            }
        }

        /// <summary>
        /// Disposes the process group and releases resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes resources.
        /// </summary>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    Destroy();
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer.
        /// </summary>
        ~ProcessGroup()
        {
            Dispose(false);
        }

        /// <summary>
        /// Gets integer environment variable or default value.
        /// </summary>
        private static int GetEnvVar(string name, int defaultValue)
        {
            var value = Environment.GetEnvironmentVariable(name);
            if (int.TryParse(value, out int result))
            {
                return result;
            }
            return defaultValue;
        }

        /// <summary>
        /// Ensures the process group has not been disposed.
        /// </summary>
        private void EnsureNotDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(ProcessGroup), "Process group has been disposed.");
            }
        }
    }
}
