using MLFramework.Core;
using RitterFramework.Core.Tensor;
using System;
using System.Threading.Tasks;

namespace MLFramework.Distributed.Gloo
{
    /// <summary>
    /// Process group implementation using Gloo backend.
    /// Gloo provides CPU and multi-GPU communication, and works on both Linux and Windows.
    /// </summary>
    public class GlooProcessGroup : IProcessGroup
    {
        private readonly GlooBackend _backend;
        private readonly GlooAllReduce _allReduce;
        private bool _disposed;

        public GlooProcessGroup(GlooBackend backend)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            backend.Initialize();

            _allReduce = new GlooAllReduce(backend.Context, backend.Rank, backend.Name);
            _disposed = false;
        }

        /// <summary>
        /// Gets the rank of this process in the process group.
        /// </summary>
        public int Rank => _backend.Rank;

        /// <summary>
        /// Gets the total number of processes in the process group.
        /// </summary>
        public int WorldSize => _backend.WorldSize;

        /// <summary>
        /// Gets the communication backend used by this process group.
        /// </summary>
        public ICommunicationBackend Backend => _backend;

        /// <summary>
        /// Performs an AllReduce operation on the given tensor.
        /// </summary>
        public void AllReduce(Tensor tensor, ReduceOp op = ReduceOp.Sum)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            var device = tensor.GetDevice();

            if (device.Type == DeviceType.CPU)
            {
                AllReduceCPU(tensor, op);
            }
            else if (device.Type == DeviceType.CUDA)
            {
                AllReduceCUDA(tensor, op);
            }
            else
            {
                throw new ArgumentException($"Unsupported device type: {device.Type}", nameof(tensor));
            }
        }

        /// <summary>
        /// Performs a Broadcast operation, sending data from root to all processes.
        /// </summary>
        public void Broadcast(Tensor tensor, int root = 0)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            if (root < 0 || root >= WorldSize)
            {
                throw new ArgumentOutOfRangeException(nameof(root), $"Root must be between 0 and {WorldSize - 1}");
            }

            var device = tensor.GetDevice();

            if (device.Type == DeviceType.CPU)
            {
                BroadcastCPU(tensor, root);
            }
            else if (device.Type == DeviceType.CUDA)
            {
                BroadcastCUDA(tensor, root);
            }
            else
            {
                throw new ArgumentException($"Unsupported device type: {device.Type}", nameof(tensor));
            }
        }

        /// <summary>
        /// Synchronizes all processes in the process group.
        /// </summary>
        public void Barrier()
        {
            try
            {
                GlooNative.gloo_barrier(_backend.Context);
            }
            catch (DllNotFoundException ex)
            {
                throw new CommunicationException("Gloo library not found", Rank, Backend.Name, ex);
            }
            catch (Exception ex)
            {
                throw new CommunicationException($"Barrier failed: {ex.Message}", Rank, Backend.Name, ex);
            }
        }

        /// <summary>
        /// Performs an asynchronous AllReduce operation.
        /// Gloo's async support is limited, so we wrap sync operation in a Task.
        /// </summary>
        public Task AllReduceAsync(Tensor tensor, ReduceOp op = ReduceOp.Sum)
        {
            return Task.Run(() => AllReduce(tensor, op));
        }

        /// <summary>
        /// Performs an asynchronous Broadcast operation.
        /// Gloo's async support is limited, so we wrap sync operation in a Task.
        /// </summary>
        public Task BroadcastAsync(Tensor tensor, int root = 0)
        {
            return Task.Run(() => Broadcast(tensor, root));
        }

        /// <summary>
        /// Performs an asynchronous barrier.
        /// Gloo's async support is limited, so we wrap sync operation in a Task.
        /// </summary>
        public Task BarrierAsync()
        {
            return Task.Run(() => Barrier());
        }

        /// <summary>
        /// Sends a tensor to the specified destination rank.
        /// </summary>
        public void Send(Tensor tensor, int dst)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            if (dst < 0 || dst >= WorldSize)
            {
                throw new ArgumentOutOfRangeException(nameof(dst), $"Destination rank must be between 0 and {WorldSize - 1}");
            }

            var device = tensor.GetDevice();

            if (device.Type == DeviceType.CPU)
            {
                SendCPU(tensor, dst);
            }
            else if (device.Type == DeviceType.CUDA)
            {
                SendCUDA(tensor, dst);
            }
            else
            {
                throw new ArgumentException($"Unsupported device type: {device.Type}", nameof(tensor));
            }
        }

        /// <summary>
        /// Receives a tensor from the specified source rank.
        /// </summary>
        public void Recv(Tensor tensor, int src)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            if (src < 0 || src >= WorldSize)
            {
                throw new ArgumentOutOfRangeException(nameof(src), $"Source rank must be between 0 and {WorldSize - 1}");
            }

            var device = tensor.GetDevice();

            if (device.Type == DeviceType.CPU)
            {
                RecvCPU(tensor, src);
            }
            else if (device.Type == DeviceType.CUDA)
            {
                RecvCUDA(tensor, src);
            }
            else
            {
                throw new ArgumentException($"Unsupported device type: {device.Type}", nameof(tensor));
            }
        }

        /// <summary>
        /// Asynchronously sends a tensor to the specified destination rank.
        /// </summary>
        public Task SendAsync(Tensor tensor, int dst)
        {
            return Task.Run(() => Send(tensor, dst));
        }

        /// <summary>
        /// Asynchronously receives a tensor from the specified source rank.
        /// </summary>
        public Task RecvAsync(Tensor tensor, int src)
        {
            return Task.Run(() => Recv(tensor, src));
        }

        /// <summary>
        /// Destroys the process group and releases resources.
        /// </summary>
        public void Destroy()
        {
            if (!_disposed)
            {
                _backend.Dispose();
                _disposed = true;
            }
        }

        private void AllReduceCPU(Tensor tensor, ReduceOp op)
        {
            _allReduce.AllReduceCPU(tensor, op);
        }

        private void AllReduceCUDA(Tensor tensor, ReduceOp op)
        {
            _allReduce.AllReduceCUDA(tensor, op);
        }

        private void BroadcastCPU(Tensor tensor, int root)
        {
            _allReduce.BroadcastCPU(tensor, root);
        }

        private void BroadcastCUDA(Tensor tensor, int root)
        {
            _allReduce.BroadcastCUDA(tensor, root);
        }

        private void SendCPU(Tensor tensor, int dst)
        {
            _allReduce.SendCPU(tensor, dst);
        }

        private void SendCUDA(Tensor tensor, int dst)
        {
            _allReduce.SendCUDA(tensor, dst);
        }

        private void RecvCPU(Tensor tensor, int src)
        {
            _allReduce.RecvCPU(tensor, src);
        }

        private void RecvCUDA(Tensor tensor, int src)
        {
            _allReduce.RecvCUDA(tensor, src);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

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

        ~GlooProcessGroup()
        {
            Dispose(false);
        }
    }
}
