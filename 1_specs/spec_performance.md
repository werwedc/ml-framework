# Spec: Performance Optimizations

## Overview
Implement performance optimizations for communication primitives including pinned memory, GPU-direct transfers, and algorithm selection.

## Dependencies
- `spec_communication_interfaces.md`
- `spec_backend_nccl.md`
- `spec_backend_mpi.md`
- `spec_backend_rccl.md`

## Technical Requirements

### 1. Memory Pinning Manager
Manage pinned memory for efficient CPU-GPU transfers.

```csharp
namespace MLFramework.Communication.Optimizations
{
    /// <summary>
    /// Manages pinned memory allocations for efficient transfers
    /// </summary>
    public class PinnedMemoryManager : IDisposable
    {
        private readonly Dictionary<IntPtr, int> _allocations;
        private readonly object _lock;
        private readonly int _maxPinnedBytes;
        private int _totalPinnedBytes;
        private bool _disposed;

        /// <summary>
        /// Maximum number of bytes that can be pinned
        /// </summary>
        public int TotalPinnedBytes => _totalPinnedBytes;

        /// <summary>
        /// Create a pinned memory manager
        /// </summary>
        /// <param name="maxPinnedBytes">Maximum bytes to pin (default: 1GB)</param>
        public PinnedMemoryManager(int maxPinnedBytes = 1024 * 1024 * 1024)
        {
            _maxPinnedBytes = maxPinnedBytes;
            _allocations = new Dictionary<IntPtr, int>();
            _lock = new object();
        }

        /// <summary>
        /// Pin memory for a tensor
        /// </summary>
        /// <returns>Handle to pinned memory</returns>
        public PinnedMemoryHandle PinMemory<T>(Tensor<T> tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            var size = GetTensorSize(tensor);

            lock (_lock)
            {
                if (_totalPinnedBytes + size > _maxPinnedBytes)
                {
                    // Try to free some memory
                    if (!TryFreeMemory(size))
                    {
                        throw new CommunicationException("Cannot pin memory: limit exceeded");
                    }
                }

                var ptr = PinMemoryInternal(tensor.Data, size);
                _allocations[ptr] = size;
                _totalPinnedBytes += size;

                return new PinnedMemoryHandle(ptr, size, this);
            }
        }

        /// <summary>
        /// Unpin memory
        /// </summary>
        internal void UnpinMemory(IntPtr ptr)
        {
            lock (_lock)
            {
                if (_allocations.TryGetValue(ptr, out int size))
                {
                    UnpinMemoryInternal(ptr);
                    _allocations.Remove(ptr);
                    _totalPinnedBytes -= size;
                }
            }
        }

        /// <summary>
        /// Try to free memory to make space
        /// </summary>
        private bool TryFreeMemory(int requiredBytes)
        {
            // Simple LRU: free oldest allocations
            foreach (var kvp in _allocations.ToList())
            {
                if (_totalPinnedBytes + requiredBytes <= _maxPinnedBytes)
                {
                    break;
                }

                UnpinMemoryInternal(kvp.Key);
                _allocations.Remove(kvp.Key);
                _totalPinnedBytes -= kvp.Value;
            }

            return _totalPinnedBytes + requiredBytes <= _maxPinnedBytes;
        }

        private IntPtr PinMemoryInternal(IntPtr data, int size)
        {
            // P/Invoke to lock pages in memory
            // Placeholder for actual implementation
            return data;
        }

        private void UnpinMemoryInternal(IntPtr ptr)
        {
            // P/Invoke to unlock pages
        }

        private int GetTensorSize<T>(Tensor<T> tensor)
        {
            return (int)(tensor.Shape.TotalSize * Marshal.SizeOf<T>());
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                lock (_lock)
                {
                    foreach (var ptr in _allocations.Keys.ToList())
                    {
                        UnpinMemoryInternal(ptr);
                    }
                    _allocations.Clear();
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Handle to pinned memory
    /// </summary>
    public class PinnedMemoryHandle : IDisposable
    {
        private readonly IntPtr _ptr;
        private readonly int _size;
        private readonly PinnedMemoryManager _manager;
        private bool _disposed;

        public IntPtr Pointer => _ptr;
        public int Size => _size;

        internal PinnedMemoryHandle(IntPtr ptr, int size, PinnedMemoryManager manager)
        {
            _ptr = ptr;
            _size = size;
            _manager = manager;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _manager.UnpinMemory(_ptr);
                _disposed = true;
            }
        }
    }
}
```

### 2. GPU-Direct Transfer Support
Enable direct GPU-to-GPU transfers bypassing CPU.

```csharp
namespace MLFramework.Communication.Optimizations
{
    /// <summary>
    /// Manages GPU-direct transfers
    /// </summary>
    public static class GPUDirectManager
    {
        private static readonly Lazy<bool> _isSupported = new Lazy<bool>(CheckGPUDirectSupport);

        /// <summary>
        /// Check if GPU-direct is supported
        /// </summary>
        public static bool IsSupported => _isSupported.Value;

        /// <summary>
        /// Check if GPU-direct is supported
        /// </summary>
        private static bool CheckGPUDirectSupport()
        {
            try
            {
                // Check for RDMA-capable NIC
                // Check for GPU-direct enabled in driver
                return false; // Placeholder
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Enable GPU-direct for a communication operation
        /// </summary>
        public static void EnableGPUDirect(ICommunicationBackend backend)
        {
            if (!IsSupported)
            {
                throw new CommunicationException("GPU-direct is not supported on this system");
            }

            // Configure backend to use GPU-direct
            if (backend is NCCLBackend nccl)
            {
                // Enable NCCL GPU-direct
                NCCLConfig.SetEnvironmentVariable("NCCL_IB_DISABLE", "0");
            }
            else if (backend is RCCLBackend rccl)
            {
                // Enable RCCL GPU-direct
                RCCLConfig.SetEnvironmentVariable("RCCL_IB_DISABLE", "0");
            }
        }
    }
}
```

### 3. Communication Algorithm Selector
Automatically select optimal algorithm based on data size and topology.

```csharp
namespace MLFramework.Communication.Optimizations
{
    /// <summary>
    /// Communication algorithms
    /// </summary>
    public enum CommunicationAlgorithm
    {
        Ring,
        Tree,
        RecursiveDoubling,
        Rabenseifner,
        Automatic
    }

    /// <summary>
    /// Selects optimal communication algorithm
    /// </summary>
    public class AlgorithmSelector
    {
        private readonly int _worldSize;
        private readonly CommunicationConfig _config;

        public AlgorithmSelector(int worldSize, CommunicationConfig config)
        {
            _worldSize = worldSize;
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Select optimal algorithm for all-reduce
        /// </summary>
        public CommunicationAlgorithm SelectAllReduceAlgorithm(long dataSizeBytes)
        {
            // Automatic selection based on data size and world size

            // For small messages, use recursive doubling
            if (dataSizeBytes < 4096 && _worldSize <= 8)
            {
                return CommunicationAlgorithm.RecursiveDoubling;
            }

            // For medium messages, use Rabenseifner
            if (dataSizeBytes < 1024 * 1024 && _worldSize <= 16)
            {
                return CommunicationAlgorithm.Rabenseifner;
            }

            // For large messages, use ring
            if (dataSizeBytes < 16 * 1024 * 1024)
            {
                return CommunicationAlgorithm.Ring;
            }

            // For very large messages, use tree
            return CommunicationAlgorithm.Tree;
        }

        /// <summary>
        /// Select optimal algorithm for all-gather
        /// </summary>
        public CommunicationAlgorithm SelectAllGatherAlgorithm(long dataSizeBytes)
        {
            // Ring is generally optimal for all-gather
            return CommunicationAlgorithm.Ring;
        }

        /// <summary>
        /// Select optimal algorithm for reduce-scatter
        /// </summary>
        public CommunicationAlgorithm SelectReduceScatterAlgorithm(long dataSizeBytes)
        {
            // Rabenseifner is optimal for reduce-scatter
            return CommunicationAlgorithm.Rabenseifner;
        }

        /// <summary>
        /// Get algorithm name
        /// </summary>
        public static string GetAlgorithmName(CommunicationAlgorithm algorithm)
        {
            return algorithm switch
            {
                CommunicationAlgorithm.Ring => "Ring",
                CommunicationAlgorithm.Tree => "Tree",
                CommunicationAlgorithm.RecursiveDoubling => "RecursiveDoubling",
                CommunicationAlgorithm.Rabenseifner => "Rabenseifner",
                CommunicationAlgorithm.Automatic => "Automatic",
                _ => "Unknown"
            };
        }
    }
}
```

### 4. Communication Profiler
Profile communication operations for performance analysis.

```csharp
namespace MLFramework.Communication.Optimizations
{
    /// <summary>
    /// Profile data for a communication operation
    /// </summary>
    public class CommunicationProfile
    {
        public string Operation { get; set; }
        public long DataSizeBytes { get; set; }
        public TimeSpan Duration { get; set; }
        public double BandwidthMBps { get; set; }
        public int NumRanks { get; set; }
        public string Algorithm { get; set; }
        public DateTime Timestamp { get; set; }

        public override string ToString()
        {
            return $"{Operation} ({Algorithm}): {DataSizeBytes / 1024.0 / 1024.0:F2} MB, " +
                   $"{Duration.TotalMilliseconds:F2} ms, " +
                   $"{BandwidthMBps:F2} MB/s, " +
                   $"{NumRanks} ranks";
        }
    }

    /// <summary>
    /// Profiler for communication operations
    /// </summary>
    public class CommunicationProfiler : IDisposable
    {
        private readonly List<CommunicationProfile> _profiles;
        private readonly object _lock;
        private readonly bool _enabled;
        private bool _disposed;

        public IReadOnlyList<CommunicationProfile> Profiles
        {
            get
            {
                lock (_lock)
                {
                    return _profiles.ToList();
                }
            }
        }

        public CommunicationProfiler(bool enabled = true)
        {
            _enabled = enabled;
            _profiles = new List<CommunicationProfile>();
            _lock = new object();
        }

        /// <summary>
        /// Profile a communication operation
        /// </summary>
        public T Profile<T>(string operation, long dataSizeBytes, Func<T> func, int numRanks = 0, string algorithm = "")
        {
            if (!_enabled)
            {
                return func();
            }

            var stopwatch = Stopwatch.StartNew();
            var result = func();
            stopwatch.Stop();

            var profile = new CommunicationProfile
            {
                Operation = operation,
                DataSizeBytes = dataSizeBytes,
                Duration = stopwatch.Elapsed,
                BandwidthMBps = CalculateBandwidth(dataSizeBytes, stopwatch.Elapsed),
                NumRanks = numRanks,
                Algorithm = algorithm,
                Timestamp = DateTime.Now
            };

            lock (_lock)
            {
                _profiles.Add(profile);
            }

            return result;
        }

        /// <summary>
        /// Profile async operation
        /// </summary>
        public async Task<T> ProfileAsync<T>(string operation, long dataSizeBytes, Func<Task<T>> func, int numRanks = 0, string algorithm = "")
        {
            if (!_enabled)
            {
                return await func();
            }

            var stopwatch = Stopwatch.StartNew();
            var result = await func();
            stopwatch.Stop();

            var profile = new CommunicationProfile
            {
                Operation = operation,
                DataSizeBytes = dataSizeBytes,
                Duration = stopwatch.Elapsed,
                BandwidthMBps = CalculateBandwidth(dataSizeBytes, stopwatch.Elapsed),
                NumRanks = numRanks,
                Algorithm = algorithm,
                Timestamp = DateTime.Now
            };

            lock (_lock)
            {
                _profiles.Add(profile);
            }

            return result;
        }

        private double CalculateBandwidth(long dataSizeBytes, TimeSpan duration)
        {
            if (duration.TotalSeconds == 0)
                return 0;

            return (dataSizeBytes / 1024.0 / 1024.0) / duration.TotalSeconds;
        }

        /// <summary>
        /// Clear all profiles
        /// </summary>
        public void Clear()
        {
            lock (_lock)
            {
                _profiles.Clear();
            }
        }

        /// <summary>
        /// Get statistics
        /// </summary>
        public CommunicationProfileStatistics GetStatistics()
        {
            lock (_lock)
            {
                if (_profiles.Count == 0)
                {
                    return new CommunicationProfileStatistics();
                }

                return new CommunicationProfileStatistics
                {
                    TotalOperations = _profiles.Count,
                    TotalDataTransferred = _profiles.Sum(p => p.DataSizeBytes),
                    TotalTime = _profiles.Sum(p => p.Duration.TotalMilliseconds),
                    AverageBandwidth = _profiles.Average(p => p.BandwidthMBps),
                    MinBandwidth = _profiles.Min(p => p.BandwidthMBps),
                    MaxBandwidth = _profiles.Max(p => p.BandwidthMBps)
                };
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                lock (_lock)
                {
                    _profiles.Clear();
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Statistics for communication profiles
    /// </summary>
    public class CommunicationProfileStatistics
    {
        public int TotalOperations { get; set; }
        public long TotalDataTransferred { get; set; }
        public double TotalTime { get; set; }
        public double AverageBandwidth { get; set; }
        public double MinBandwidth { get; set; }
        public double MaxBandwidth { get; set; }

        public override string ToString()
        {
            return $"Total Operations: {TotalOperations}, " +
                   $"Total Data: {TotalDataTransferred / 1024.0 / 1024.0 / 1024.0:F2} GB, " +
                   $"Total Time: {TotalTime / 1000.0:F2} s, " +
                   $"Avg Bandwidth: {AverageBandwidth:F2} MB/s";
        }
    }
}
```

### 5. Communication Optimizer
High-level optimizer that applies multiple optimizations.

```csharp
namespace MLFramework.Communication.Optimizations
{
    /// <summary>
    /// High-level optimizer for communication operations
    /// </summary>
    public class CommunicationOptimizer
    {
        private readonly ICommunicationBackend _backend;
        private readonly PinnedMemoryManager? _pinnedMemoryManager;
        private readonly AlgorithmSelector _algorithmSelector;
        private readonly CommunicationProfiler _profiler;

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
        public Tensor<T> AllReduceOptimized<T>(
            Tensor<T> tensor,
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
        public ICommunicationHandle AllReduceOptimizedAsync<T>(
            Tensor<T> tensor,
            ReduceOp operation)
        {
            var dataSizeBytes = GetTensorSize(tensor);
            var algorithm = _algorithmSelector.SelectAllReduceAlgorithm(dataSizeBytes);

            if (_backend is IAsyncCommunicationBackend asyncBackend)
            {
                var task = Task.Run(() => _backend.AllReduce(tensor, operation));
                var handle = new AsyncCommunicationHandle(task);

                // Profile in background
                Task.Run(() => _profiler.ProfileAsync(
                    "AllReduceAsync",
                    dataSizeBytes,
                    () => handle.AsTask(),
                    _backend.WorldSize,
                    AlgorithmSelector.GetAlgorithmName(algorithm)
                ));

                return handle;
            }

            throw new NotSupportedException("Backend does not support async operations");
        }

        private long GetTensorSize<T>(Tensor<T> tensor)
        {
            return tensor.Shape.TotalSize * Marshal.SizeOf<T>();
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
            _pinnedMemoryManager?.Dispose();
            _profiler.Dispose();
        }
    }
}
```

## Implementation Notes

1. **File Structure:**
   - `src/MLFramework/Communication/Optimizations/PinnedMemoryManager.cs`
   - `src/MLFramework/Communication/Optimizations/PinnedMemoryHandle.cs`
   - `src/MLFramework/Communication/Optimizations/GPUDirectManager.cs`
   - `src/MLFramework/Communication/Optimizations/AlgorithmSelector.cs`
   - `src/MLFramework/Communication/Optimizations/CommunicationProfiler.cs`
   - `src/MLFramework/Communication/Optimizations/CommunicationProfile.cs`
   - `src/MLFramework/Communication/Optimizations/CommunicationOptimizer.cs`

2. **Design Decisions:**
   - Pinned memory uses LRU eviction strategy
   - GPU-direct is auto-detected and enabled
   - Algorithm selection based on data size and topology
   - Profiler tracks all operations for analysis

3. **Error Handling:**
   - Throw exceptions when memory limits exceeded
   - Validate GPU-direct support before enabling
   - Handle profiler errors gracefully

4. **Performance Considerations:**
   - Minimize allocations in hot paths
   - Use efficient data structures for tracking
   - Async profiling to avoid overhead

## Testing Requirements
- Tests for pinned memory allocation and deallocation
- Tests for GPU-direct detection
- Tests for algorithm selection
- Tests for profiler accuracy
- Performance benchmarking tests

## Success Criteria
- Pinned memory manager correctly allocates and frees memory
- GPU-direct support is detected correctly
- Algorithm selector chooses optimal algorithms
- Profiler accurately measures performance
- Optimizer integrates all optimizations
