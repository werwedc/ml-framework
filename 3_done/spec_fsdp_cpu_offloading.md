# Spec: FSDP CPU Offloading

## Overview
Implement CPU offloading to further reduce GPU memory usage by moving parameters/gradients to CPU when not in use.

## Requirements

### 1. FSDPCpuOffloadConfig Class
Define configuration for CPU offloading:

```csharp
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
```

### 2. FSDPCpuOffloader Class
Create a class to manage CPU offloading:

```csharp
public class FSDPCpuOffloader : IDisposable
{
    private readonly FSDPCpuOffloadConfig _config;
    private readonly FSDP _fsdp;
    private readonly Dictionary<string, CpuOffloadBuffer> _cpuBuffers;

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
    }

    /// <summary>
    /// Prefetch Adam optimizer state from CPU to GPU.
    /// </summary>
    private void PrefetchAdamState(AdamOptimizerState adamState)
    {
        if (!_cpuBuffers.TryGetValue(adamState.ToString(), out var buffer))
            return;

        // Copy buffers from CPU to GPU
        CopyFromDevice(buffer.MomentumBuffer, adamState.MomentumBuffer);
        CopyFromDevice(buffer.VarianceBuffer, adamState.VarianceBuffer);
    }

    /// <summary>
    /// Create a CPU offload buffer for a sharding unit.
    /// </summary>
    private CpuOffloadBuffer CreateCpuBuffer(FSDPShardingUnit shardingUnit)
    {
        var paramSize = shardingUnit.ShardedParameter!.Size;

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
    /// Dispose of resources.
    /// </summary>
    public void Dispose()
    {
        _cpuBuffers.Clear();
    }
}
```

### 3. CpuOffloadBuffer Class
Define a buffer for CPU offloading:

```csharp
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
```

### 4. FSDPPrefetchManager Class
Create a prefetch manager for overlapping CPU-GPU transfers with computation:

```csharp
public class FSDPPrefetchManager : IDisposable
{
    private readonly FSDPCpuOffloader _offloader;
    private readonly Queue<PrefetchTask> _prefetchQueue;
    private readonly int _prefetchSteps;

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

        _prefetchQueue.Enqueue(task);
    }

    /// <summary>
    /// Execute prefetch tasks asynchronously.
    /// </summary>
    public async Task ExecutePrefetchAsync()
    {
        while (_prefetchQueue.Count > 0)
        {
            var task = _prefetchQueue.Peek();

            // Check if it's time to execute this task
            var timeUntilExecution = task.ScheduledTime.Add(TimeSpan.FromMilliseconds(10)) - DateTime.UtcNow;

            if (timeUntilExecution.TotalMilliseconds <= 0)
            {
                _prefetchQueue.Dequeue();

                // Execute prefetch
                await Task.Run(() =>
                {
                    foreach (var unit in task.ShardingUnits.Values)
                    {
                        if (task.IsForwardPass)
                        {
                            _offloader.PrefetchParameter(unit);
                        }
                        else
                        {
                            _offloader.PrefetchGradient(unit);
                        }
                    }
                });
            }
            else
            {
                await Task.Delay(timeUntilExecution);
            }
        }
    }

    /// <summary>
    /// Clear all pending prefetch tasks.
    /// </summary>
    public void Clear()
    {
        _prefetchQueue.Clear();
    }

    /// <summary>
    /// Dispose of resources.
    /// </summary>
    public void Dispose()
    {
        Clear();
    }
}

public class PrefetchTask
{
    public Dictionary<string, FSDPShardingUnit> ShardingUnits { get; set; }
    public bool IsForwardPass { get; set; }
    public DateTime ScheduledTime { get; set; }
}
```

## Directory Structure
- **File**: `src/MLFramework/Distributed/FSDP/FSDPCpuOffload.cs`
- **Namespace**: `MLFramework.Distributed.FSDP`

## Dependencies
- `MLFramework.Distributed.FSDP.FSDP`
- `MLFramework.Distributed.FSDP.FSDPShardingUnit`
- `MLFramework.Distributed.FSDP.OptimizerState`
- `RitterFramework.Core.Tensor`

## Implementation Notes
1. Offload parameters/gradients to CPU when not in use
2. Prefetch data before it's needed to overlap with computation
3. Manage CPU buffers for efficient memory usage
4. Handle both forward pass (parameters) and backward pass (gradients)
5. Support optimizer state offloading

## Testing Requirements
- Test parameter offloading to CPU
- Test gradient offloading to CPU
- Test optimizer state offloading
- Test parameter prefetching from CPU
- Test gradient prefetching from CPU
- Test prefetch scheduling
- Test edge cases (no offloading, single device)

## Estimated Time
60 minutes
