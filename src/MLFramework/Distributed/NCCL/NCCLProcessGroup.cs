using MLFramework.Distributed;
using MLFramework.Distributed.NCCL;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace MLFramework.Distributed.NCCL
{
    /// <summary>
    /// Process group implementation using NCCL backend.
    /// </summary>
    public class NCCLProcessGroup : IProcessGroup
    {
        private readonly NCCLBackend _backend;
        private bool _disposed;

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

        public NCCLProcessGroup(NCCLBackend backend)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            _disposed = false;

            _backend.Initialize();
        }

        /// <summary>
        /// Performs an AllReduce operation on the given tensor.
        /// </summary>
        public void AllReduce(Tensor tensor, ReduceOp op = ReduceOp.Sum)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!_backend.Initialized)
                throw new InvalidOperationException("NCCL backend is not initialized");

            // Pin the tensor data and get pointer
            var handle = GCHandle.Alloc(tensor.Data, GCHandleType.Pinned);
            try
            {
                var tensorPtr = handle.AddrOfPinnedObject();
                var numElements = (ulong)tensor.Size;
                var dataType = GetNCCLDataType(tensor.Dtype);
                var ncclOp = GetNCCLOp(op);

                var error = NCCLNative.ncclAllReduce(
                    tensorPtr,
                    tensorPtr,
                    numElements,
                    dataType,
                    ncclOp,
                    _backend.Comm,
                    IntPtr.Zero); // Default stream

                NCCLNative.CheckError(error, Rank, "ncclAllReduce");
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Performs an asynchronous AllReduce operation.
        /// </summary>
        public Task AllReduceAsync(Tensor tensor, ReduceOp op = ReduceOp.Sum)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!_backend.Initialized)
                throw new InvalidOperationException("NCCL backend is not initialized");

            // Pin the tensor data and get pointer
            var handle = GCHandle.Alloc(tensor.Data, GCHandleType.Pinned);
            try
            {
                var tensorPtr = handle.AddrOfPinnedObject();
                var numElements = (ulong)tensor.Size;
                var dataType = GetNCCLDataType(tensor.Dtype);
                var ncclOp = GetNCCLOp(op);

                // For async operations, we would use a CUDA stream
                // For now, we'll use IntPtr.Zero (default stream)
                // TODO: Implement proper CUDA stream management when CUDA integration is available
                var streamPtr = IntPtr.Zero;

                var error = NCCLNative.ncclAllReduce(
                    tensorPtr,
                    tensorPtr,
                    numElements,
                    dataType,
                    ncclOp,
                    _backend.Comm,
                    streamPtr);

                NCCLNative.CheckError(error, Rank, "ncclAllReduce");

                // Return a completed task for now
                // In a real implementation, this would wait for the CUDA stream to complete
                return Task.CompletedTask;
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Performs a Broadcast operation, sending data from root to all processes.
        /// </summary>
        public void Broadcast(Tensor tensor, int root = 0)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!_backend.Initialized)
                throw new InvalidOperationException("NCCL backend is not initialized");

            if (root < 0 || root >= WorldSize)
                throw new ArgumentException($"Root rank {root} is out of bounds (0-{WorldSize - 1})");

            // Pin the tensor data and get pointer
            var handle = GCHandle.Alloc(tensor.Data, GCHandleType.Pinned);
            try
            {
                var tensorPtr = handle.AddrOfPinnedObject();
                var numElements = (ulong)tensor.Size;
                var dataType = GetNCCLDataType(tensor.Dtype);

                var error = NCCLNative.ncclBroadcast(
                    tensorPtr,
                    tensorPtr,
                    numElements,
                    dataType,
                    root,
                    _backend.Comm,
                    IntPtr.Zero); // Default stream

                NCCLNative.CheckError(error, Rank, "ncclBroadcast");
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Performs an asynchronous Broadcast operation.
        /// </summary>
        public Task BroadcastAsync(Tensor tensor, int root = 0)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!_backend.Initialized)
                throw new InvalidOperationException("NCCL backend is not initialized");

            if (root < 0 || root >= WorldSize)
                throw new ArgumentException($"Root rank {root} is out of bounds (0-{WorldSize - 1})");

            // Pin the tensor data and get pointer
            var handle = GCHandle.Alloc(tensor.Data, GCHandleType.Pinned);
            try
            {
                var tensorPtr = handle.AddrOfPinnedObject();
                var numElements = (ulong)tensor.Size;
                var dataType = GetNCCLDataType(tensor.Dtype);

                // For async operations, we would use a CUDA stream
                // For now, we'll use IntPtr.Zero (default stream)
                var streamPtr = IntPtr.Zero;

                var error = NCCLNative.ncclBroadcast(
                    tensorPtr,
                    tensorPtr,
                    numElements,
                    dataType,
                    root,
                    _backend.Comm,
                    streamPtr);

                NCCLNative.CheckError(error, Rank, "ncclBroadcast");

                // Return a completed task for now
                // In a real implementation, this would wait for the CUDA stream to complete
                return Task.CompletedTask;
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Synchronizes all processes in the process group.
        /// </summary>
        public void Barrier()
        {
            if (!_backend.Initialized)
                throw new InvalidOperationException("NCCL backend is not initialized");

            var error = NCCLNative.ncclBarrier(
                _backend.Comm,
                IntPtr.Zero); // Default stream

            NCCLNative.CheckError(error, Rank, "ncclBarrier");
        }

        /// <summary>
        /// Performs an asynchronous barrier.
        /// </summary>
        public Task BarrierAsync()
        {
            if (!_backend.Initialized)
                throw new InvalidOperationException("NCCL backend is not initialized");

            var streamPtr = IntPtr.Zero; // Default stream

            var error = NCCLNative.ncclBarrier(
                _backend.Comm,
                streamPtr);

            NCCLNative.CheckError(error, Rank, "ncclBarrier");

            // Return a completed task for now
            // In a real implementation, this would wait for the CUDA stream to complete
            return Task.CompletedTask;
        }

        /// <summary>
        /// Sends a tensor to the specified destination rank.
        /// </summary>
        public void Send(Tensor tensor, int dst)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!_backend.Initialized)
                throw new InvalidOperationException("NCCL backend is not initialized");

            if (dst < 0 || dst >= WorldSize)
                throw new ArgumentException($"Destination rank {dst} is out of bounds (0-{WorldSize - 1})");

            var handle = GCHandle.Alloc(tensor.Data, GCHandleType.Pinned);
            try
            {
                var tensorPtr = handle.AddrOfPinnedObject();
                var numElements = (ulong)tensor.Size;
                var dataType = GetNCCLDataType(tensor.Dtype);

                var error = NCCLNative.ncclSend(
                    tensorPtr,
                    numElements,
                    dataType,
                    dst,
                    _backend.Comm,
                    IntPtr.Zero); // Default stream

                NCCLNative.CheckError(error, Rank, "ncclSend");
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Receives a tensor from the specified source rank.
        /// </summary>
        public void Recv(Tensor tensor, int src)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!_backend.Initialized)
                throw new InvalidOperationException("NCCL backend is not initialized");

            if (src < 0 || src >= WorldSize)
                throw new ArgumentException($"Source rank {src} is out of bounds (0-{WorldSize - 1})");

            var handle = GCHandle.Alloc(tensor.Data, GCHandleType.Pinned);
            try
            {
                var tensorPtr = handle.AddrOfPinnedObject();
                var numElements = (ulong)tensor.Size;
                var dataType = GetNCCLDataType(tensor.Dtype);

                var error = NCCLNative.ncclRecv(
                    tensorPtr,
                    numElements,
                    dataType,
                    src,
                    _backend.Comm,
                    IntPtr.Zero); // Default stream

                NCCLNative.CheckError(error, Rank, "ncclRecv");
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Asynchronously sends a tensor to the specified destination rank.
        /// </summary>
        public Task SendAsync(Tensor tensor, int dst)
        {
            Send(tensor, dst);
            return Task.CompletedTask;
        }

        /// <summary>
        /// Asynchronously receives a tensor from the specified source rank.
        /// </summary>
        public Task RecvAsync(Tensor tensor, int src)
        {
            Recv(tensor, src);
            return Task.CompletedTask;
        }

        /// <summary>
        /// Destroys the process group and releases resources.
        /// </summary>
        public void Destroy()
        {
            _backend.Cleanup();
        }

        /// <summary>
        /// Maps DataType to NCCL data type enum.
        /// </summary>
        private ncclDataType_t GetNCCLDataType(DataType dtype)
        {
            return dtype switch
            {
                DataType.Float32 => ncclDataType_t.ncclFloat32,
                DataType.Float64 => ncclDataType_t.ncclFloat64,
                DataType.Int32 => ncclDataType_t.ncclInt32,
                DataType.Int64 => ncclDataType_t.ncclInt64,
                DataType.Int16 => ncclDataType_t.ncclInt32, // NCCL doesn't support Int16, treat as Int32
                DataType.Int8 => ncclDataType_t.ncclInt8,
                DataType.UInt8 => ncclDataType_t.ncclUint8,
                DataType.Float16 => ncclDataType_t.ncclFloat16,
                DataType.Bool => ncclDataType_t.ncclInt8, // Treat bool as int8
                _ => throw new ArgumentException($"Unsupported dtype: {dtype}")
            };
        }

        /// <summary>
        /// Maps ReduceOp to NCCL reduction operation enum.
        /// </summary>
        private ncclRedOp_t GetNCCLOp(ReduceOp op)
        {
            return op switch
            {
                ReduceOp.Sum => ncclRedOp_t.ncclSum,
                ReduceOp.Avg => ncclRedOp_t.ncclAvg,
                ReduceOp.Max => ncclRedOp_t.ncclMax,
                ReduceOp.Min => ncclRedOp_t.ncclMin,
                ReduceOp.Product => ncclRedOp_t.ncclProd,
                _ => throw new ArgumentException($"Unsupported op: {op}")
            };
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

        ~NCCLProcessGroup()
        {
            Dispose(false);
        }
    }
}
