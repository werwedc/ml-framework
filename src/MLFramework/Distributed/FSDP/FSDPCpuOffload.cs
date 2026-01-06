using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Configuration for CPU offloading in FSDP.
    /// </summary>
    public class FSDPCpuOffloadConfig
    {
        /// <summary>Whether CPU offloading is enabled</summary>
        public bool Enabled { get; set; } = true;

        /// <summary>Offload optimizer states to CPU</summary>
        public bool OffloadOptimizerStates { get; set; } = true;

        /// <summary>Offload gradients to CPU after backward pass</summary>
        public bool OffloadGradients { get; set; } = true;

        /// <summary>Offload parameters to CPU after forward pass</summary>
        public bool OffloadParameters { get; set; } = true;

        /// <summary>Prefetch parameters before forward pass</summary>
        public bool PrefetchParameters { get; set; } = true;

        /// <summary>Prefetch gradients before optimizer step</summary>
        public bool PrefetchGradients { get; set; } = true;

        /// <summary>Number of steps ahead to prefetch</summary>
        public int PrefetchSteps { get; set; } = 1;

        /// <summary>Validate configuration</summary>
        public void Validate()
        {
            if (PrefetchSteps < 0 || PrefetchSteps > 10)
            {
                throw new ArgumentException("PrefetchSteps must be between 0 and 10", nameof(PrefetchSteps));
            }
        }
    }

    /// <summary>
    /// Buffer for CPU offloading.
    /// </summary>
    public class CpuOffloadBuffer
    {
        /// <summary>Parameter buffer on CPU</summary>
        public float[] ParameterBuffer { get; set; }

        /// <summary>Gradient buffer on CPU</summary>
        public float[] GradientBuffer { get; set; }

        /// <summary>Optimizer momentum buffer on CPU</summary>
        public float[] MomentumBuffer { get; set; }

        /// <summary>Optimizer variance buffer on CPU</summary>
        public float[] VarianceBuffer { get; set; }

        /// <summary>Last access time</summary>
        public DateTime LastAccessTime { get; set; }

        public CpuOffloadBuffer()
        {
            LastAccessTime = DateTime.UtcNow;
        }
    }

    /// <summary>
    /// Prefetch task for asynchronous CPU-GPU transfers.
    /// </summary>
    public class PrefetchTask
    {
        /// <summary>Sharding units to prefetch</summary>
        public Dictionary<string, FSDPShardingUnit> ShardingUnits { get; set; }

        /// <summary>Whether this is for forward pass (true) or backward pass (false)</summary>
        public bool IsForwardPass { get; set; }

        /// <summary>When this task was scheduled</summary>
        public DateTime ScheduledTime { get; set; }
    }

    /// <summary>
    /// CPU offloader for FSDP to move parameters/gradients to CPU when not in use.
    /// </summary>
    public class FSDPCpuOffloader : IDisposable
    {
        private readonly FSDPCpuOffloadConfig _config;
        private readonly FSDP _fsdp;
        private readonly Dictionary<string, CpuOffloadBuffer> _cpuBuffers;
        private bool _disposed;

        /// <summary>
        /// Initialize CPU offloader for FSDP.
        /// </summary>
        /// <param name="config">CPU offloading configuration</param>
        /// <param name="fsdp">FSDP wrapper instance</param>
        public FSDPCpuOffloader(FSDPCpuOffloadConfig config, FSDP fsdp)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _fsdp = fsdp ?? throw new ArgumentNullException(nameof(fsdp));

            _config.Validate();
            _cpuBuffers = new Dictionary<string, CpuOffloadBuffer>();
        }

        /// <summary>
        /// Offload a parameter to CPU.
        /// </summary>
        /// <param name="shardingUnit">Sharding unit with parameter to offload</param>
        public void OffloadParameter(FSDPShardingUnit shardingUnit)
        {
            if (shardingUnit == null)
                throw new ArgumentNullException(nameof(shardingUnit));

            if (!_config.OffloadParameters)
                return;

            if (shardingUnit.State.IsOffloaded)
                return; // Already offloaded

            var paramName = shardingUnit.ParameterName;

            // Get or create CPU buffer
            if (!_cpuBuffers.TryGetValue(paramName, out var buffer))
            {
                buffer = CreateCpuBuffer(shardingUnit);
                _cpuBuffers[paramName] = buffer;
            }

            // Copy parameter to CPU
            CopyToDevice(shardingUnit.ShardedParameter, buffer.ParameterBuffer);

            // Mark as offloaded
            shardingUnit.State.IsOffloaded = true;
            buffer.LastAccessTime = DateTime.UtcNow;
        }

        /// <summary>
        /// Prefetch a parameter from CPU to GPU.
        /// </summary>
        /// <param name="shardingUnit">Sharding unit with parameter to prefetch</param>
        public void PrefetchParameter(FSDPShardingUnit shardingUnit)
        {
            if (shardingUnit == null)
                throw new ArgumentNullException(nameof(shardingUnit));

            if (!_config.PrefetchParameters)
                return;

            if (!shardingUnit.State.IsOffloaded)
                return; // Already on GPU

            var paramName = shardingUnit.ParameterName;

            if (!_cpuBuffers.TryGetValue(paramName, out var buffer))
                return; // No CPU buffer exists

            // Copy parameter from CPU to GPU
            CopyFromDevice(buffer.ParameterBuffer, shardingUnit.ShardedParameter);

            // Mark as on GPU
            shardingUnit.State.IsOffloaded = false;
            buffer.LastAccessTime = DateTime.UtcNow;
        }

        /// <summary>
        /// Offload a gradient to CPU.
        /// </summary>
        /// <param name="shardingUnit">Sharding unit with gradient to offload</param>
        public void OffloadGradient(FSDPShardingUnit shardingUnit)
        {
            if (shardingUnit == null)
                throw new ArgumentNullException(nameof(shardingUnit));

            if (!_config.OffloadGradients)
                return;

            if (shardingUnit.LocalGradient == null)
                return;

            var paramName = shardingUnit.ParameterName;

            // Get or create CPU buffer
            if (!_cpuBuffers.TryGetValue(paramName, out var buffer))
            {
                buffer = CreateCpuBuffer(shardingUnit);
                _cpuBuffers[paramName] = buffer;
            }

            // Copy gradient to CPU
            CopyToDevice(shardingUnit.LocalGradient, buffer.GradientBuffer);

            // Mark as offloaded
            shardingUnit.State.IsOffloaded = true;
            buffer.LastAccessTime = DateTime.UtcNow;
        }

        /// <summary>
        /// Prefetch a gradient from CPU to GPU.
        /// </summary>
        /// <param name="shardingUnit">Sharding unit with gradient to prefetch</param>
        public void PrefetchGradient(FSDPShardingUnit shardingUnit)
        {
            if (shardingUnit == null)
                throw new ArgumentNullException(nameof(shardingUnit));

            if (!_config.PrefetchGradients)
                return;

            if (shardingUnit.LocalGradient == null)
                return;

            var paramName = shardingUnit.ParameterName;

            if (!_cpuBuffers.TryGetValue(paramName, out var buffer))
                return; // No CPU buffer exists

            // Copy gradient from CPU to GPU
            CopyFromDevice(buffer.GradientBuffer, shardingUnit.LocalGradient);

            // Mark as on GPU
            shardingUnit.State.IsOffloaded = false;
            buffer.LastAccessTime = DateTime.UtcNow;
        }

        /// <summary>
        /// Offload optimizer states to CPU.
        /// </summary>
        /// <param name="optimizerState">Optimizer state to offload</param>
        public void OffloadOptimizerState(OptimizerState optimizerState)
        {
            if (optimizerState == null)
                throw new ArgumentNullException(nameof(optimizerState));

            if (!_config.OffloadOptimizerStates)
                return;

            if (optimizerState is AdamOptimizerState adamState)
            {
                OffloadAdamState(adamState);
            }
        }

        /// <summary>
        /// Prefetch optimizer states from CPU to GPU.
        /// </summary>
        /// <param name="optimizerState">Optimizer state to prefetch</param>
        public void PrefetchOptimizerState(OptimizerState optimizerState)
        {
            if (optimizerState == null)
                throw new ArgumentNullException(nameof(optimizerState));

            if (optimizerState is AdamOptimizerState adamState)
            {
                PrefetchAdamState(adamState);
            }
        }

        /// <summary>
        /// Offload Adam optimizer state to CPU.
        /// </summary>
        private void OffloadAdamState(AdamOptimizerState adamState)
        {
            if (adamState.MomentumBuffer == null || adamState.VarianceBuffer == null)
                return;

            if (!_cpuBuffers.TryGetValue(adamState.ToString(), out var buffer))
            {
                buffer = new CpuOffloadBuffer
                {
                    MomentumBuffer = new float[adamState.MomentumBuffer.Size],
                    VarianceBuffer = new float[adamState.VarianceBuffer.Size]
                };
                _cpuBuffers[adamState.ToString()] = buffer;
            }

            // Copy buffers to CPU
            CopyToDevice(adamState.MomentumBuffer, buffer.MomentumBuffer);
            CopyToDevice(adamState.VarianceBuffer, buffer.VarianceBuffer);

            buffer.LastAccessTime = DateTime.UtcNow;
        }

        /// <summary>
        /// Prefetch Adam optimizer state from CPU to GPU.
        /// </summary>
        private void PrefetchAdamState(AdamOptimizerState adamState)
        {
            if (adamState.MomentumBuffer == null || adamState.VarianceBuffer == null)
                return;

            if (!_cpuBuffers.TryGetValue(adamState.ToString(), out var buffer))
                return;

            // Copy buffers from CPU to GPU
            CopyFromDevice(buffer.MomentumBuffer, adamState.MomentumBuffer);
            CopyFromDevice(buffer.VarianceBuffer, adamState.VarianceBuffer);

            buffer.LastAccessTime = DateTime.UtcNow;
        }

        /// <summary>
        /// Create a CPU offload buffer for a sharding unit.
        /// </summary>
        private CpuOffloadBuffer CreateCpuBuffer(FSDPShardingUnit shardingUnit)
        {
            var paramSize = shardingUnit.ShardedParameter?.Size ?? 0;

            return new CpuOffloadBuffer
            {
                ParameterBuffer = new float[paramSize],
                GradientBuffer = new float[paramSize]
            };
        }

        /// <summary>
        /// Copy tensor data to CPU.
        /// </summary>
        private void CopyToDevice(Tensor tensor, float[] cpuBuffer)
        {
            if (tensor == null || cpuBuffer == null)
                return;

            Array.Copy(tensor.Data, 0, cpuBuffer, 0, Math.Min(tensor.Size, cpuBuffer.Length));
        }

        /// <summary>
        /// Copy tensor data from CPU.
        /// </summary>
        private void CopyFromDevice(float[] cpuBuffer, Tensor tensor)
        {
            if (cpuBuffer == null || tensor == null)
                return;

            Array.Copy(cpuBuffer, 0, tensor.Data, 0, Math.Min(cpuBuffer.Length, tensor.Size));
        }

        /// <summary>
        /// Get CPU buffer for a parameter name.
        /// </summary>
        public CpuOffloadBuffer GetCpuBuffer(string parameterName)
        {
            _cpuBuffers.TryGetValue(parameterName, out var buffer);
            return buffer;
        }

        /// <summary>
        /// Clear all CPU buffers.
        /// </summary>
        public void ClearBuffers()
        {
            _cpuBuffers.Clear();
        }

        /// <summary>
        /// Dispose of resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Protected implementation of dispose pattern.
        /// </summary>
        /// <param name="disposing">Whether managed resources should be disposed</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    ClearBuffers();
                }

                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for FSDPCpuOffloader.
        /// </summary>
        ~FSDPCpuOffloader()
        {
            Dispose(false);
        }
    }

    /// <summary>
    /// Prefetch manager for overlapping CPU-GPU transfers with computation.
    /// </summary>
    public class FSDPPrefetchManager : IDisposable
    {
        private readonly FSDPCpuOffloader _offloader;
        private readonly Queue<PrefetchTask> _prefetchQueue;
        private readonly int _prefetchSteps;
        private readonly object _queueLock = new object();
        private CancellationTokenSource _cancellationTokenSource;
        private Task _prefetchTask;
        private bool _disposed;

        /// <summary>
        /// Initialize prefetch manager.
        /// </summary>
        /// <param name="offloader">CPU offloader instance</param>
        /// <param name="prefetchSteps">Number of steps to prefetch ahead</param>
        public FSDPPrefetchManager(FSDPCpuOffloader offloader, int prefetchSteps = 1)
        {
            _offloader = offloader ?? throw new ArgumentNullException(nameof(offloader));
            _prefetchSteps = Math.Max(0, prefetchSteps);
            _prefetchQueue = new Queue<PrefetchTask>();
            _cancellationTokenSource = new CancellationTokenSource();
        }

        /// <summary>
        /// Schedule a prefetch task.
        /// </summary>
        /// <param name="shardingUnits">Sharding units to prefetch</param>
        /// <param name="isForwardPass">Whether this is for forward pass</param>
        public void SchedulePrefetch(Dictionary<string, FSDPShardingUnit> shardingUnits, bool isForwardPass)
        {
            if (_prefetchSteps == 0)
                return;

            var task = new PrefetchTask
            {
                ShardingUnits = shardingUnits,
                IsForwardPass = isForwardPass,
                ScheduledTime = DateTime.UtcNow
            };

            lock (_queueLock)
            {
                _prefetchQueue.Enqueue(task);

                // Start the prefetch task if not already running
                if (_prefetchTask == null || _prefetchTask.IsCompleted)
                {
                    _prefetchTask = Task.Run(() => ExecutePrefetchAsync(_cancellationTokenSource.Token));
                }
            }
        }

        /// <summary>
        /// Execute prefetch tasks asynchronously.
        /// </summary>
        private async Task ExecutePrefetchAsync(CancellationToken cancellationToken)
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                PrefetchTask task = null;

                lock (_queueLock)
                {
                    if (_prefetchQueue.Count > 0)
                    {
                        task = _prefetchQueue.Dequeue();
                    }
                }

                if (task != null)
                {
                    // Check if it's time to execute this task
                    var timeUntilExecution = task.ScheduledTime.Add(TimeSpan.FromMilliseconds(10)) - DateTime.UtcNow;

                    if (timeUntilExecution.TotalMilliseconds <= 0)
                    {
                        // Execute prefetch
                        await Task.Run(() =>
                        {
                            if (cancellationToken.IsCancellationRequested)
                                return;

                            foreach (var unit in task.ShardingUnits.Values)
                            {
                                if (cancellationToken.IsCancellationRequested)
                                    break;

                                if (task.IsForwardPass)
                                {
                                    _offloader.PrefetchParameter(unit);
                                }
                                else
                                {
                                    _offloader.PrefetchGradient(unit);
                                }
                            }
                        }, cancellationToken);
                    }
                    else if (timeUntilExecution.TotalMilliseconds > 0)
                    {
                        await Task.Delay(timeUntilExecution, cancellationToken);
                    }
                }
                else
                {
                    // No tasks in queue, wait a bit
                    await Task.Delay(100, cancellationToken);
                }
            }
        }

        /// <summary>
        /// Clear all pending prefetch tasks.
        /// </summary>
        public void Clear()
        {
            lock (_queueLock)
            {
                _prefetchQueue.Clear();
            }
        }

        /// <summary>
        /// Stop the prefetch manager.
        /// </summary>
        public void Stop()
        {
            _cancellationTokenSource?.Cancel();
            try
            {
                _prefetchTask?.Wait(TimeSpan.FromSeconds(5));
            }
            catch (AggregateException)
            {
                // Task was cancelled or faulted
            }
        }

        /// <summary>
        /// Dispose of resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Protected implementation of dispose pattern.
        /// </summary>
        /// <param name="disposing">Whether managed resources should be disposed</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    Stop();
                    Clear();
                    _cancellationTokenSource?.Dispose();
                }

                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for FSDPPrefetchManager.
        /// </summary>
        ~FSDPPrefetchManager()
        {
            Dispose(false);
        }
    }
}
