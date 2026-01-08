using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using MLFramework.Communication;
using MLFramework.Communication.Operations;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;

namespace MLFramework.Communication.Optimizations
{
    /// <summary>
    /// High-level optimizer for communication operations
    /// </summary>
    public class CommunicationOptimizer : IDisposable
    {
        private readonly ICommunicationBackend _backend;
        private readonly PinnedMemoryManager? _pinnedMemoryManager;
        private readonly AlgorithmSelector _algorithmSelector;
        private readonly CommunicationProfiler _profiler;
        private bool _disposed;

        public CommunicationOptimizer(
            ICommunicationBackend backend,
            CommunicationConfig config)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));

            if (config.UsePinnedMemory)
            {
                _pinnedMemoryManager = new PinnedMemoryManager();
            }

            _algorithmSelector = new AlgorithmSelector(backend.WorldSize, config);
            _profiler = new CommunicationProfiler(config.EnableLogging);
        }

        /// <summary>
        /// Optimized all-reduce operation
        /// </summary>
        public Tensor AllReduceOptimized(
            Tensor tensor,
            ReduceOp operation)
        {
            var dataSizeBytes = GetTensorSize(tensor);
            var algorithm = _algorithmSelector.SelectAllReduceAlgorithm(dataSizeBytes);

            return _profiler.Profile(
                "AllReduce",
                dataSizeBytes,
                () => _backend.AllReduce(tensor, operation),
                _backend.WorldSize,
                AlgorithmSelector.GetAlgorithmName(algorithm)
            );
        }

        /// <summary>
        /// Optimized async all-reduce operation
        /// </summary>
        public ICommunicationHandle AllReduceOptimizedAsync(
            Tensor tensor,
            ReduceOp operation)
        {
            var dataSizeBytes = GetTensorSize(tensor);
            var algorithm = _algorithmSelector.SelectAllReduceAlgorithm(dataSizeBytes);

            if (_backend is IAsyncCommunicationBackend asyncBackend)
            {
                var task = Task.Run(() => _backend.AllReduce(tensor, operation));
                var handle = new PendingOperationHandle(task);

                // Note: Profiling is simplified for async operations
                // The actual duration is measured when the operation completes

                return handle;
            }

            throw new NotSupportedException("Backend does not support async operations");
        }

        private long GetTensorSize(Tensor tensor)
        {
            return tensor.Size * Marshal.SizeOf<float>();
        }

        /// <summary>
        /// Get profiler
        /// </summary>
        public CommunicationProfiler Profiler => _profiler;

        /// <summary>
        /// Get algorithm selector
        /// </summary>
        public AlgorithmSelector AlgorithmSelector => _algorithmSelector;

        public void Dispose()
        {
            if (!_disposed)
            {
                _pinnedMemoryManager?.Dispose();
                _profiler.Dispose();
                _disposed = true;
            }
        }
    }
}
