# Spec: GPU Utilization Tracker

## Overview
Implement GPU utilization tracking to monitor hardware usage, identify bottlenecks, and optimize performance across available devices.

## Objectives
- Track GPU utilization percentage over time
- Monitor GPU memory usage
- Detect idle time and underutilization
- Provide insights for performance optimization

## API Design

```csharp
// GPU device information
public class GPUDeviceInfo
{
    public int DeviceId { get; }
    public string Name { get; }
    public long TotalMemoryBytes { get; }
    public int ComputeCapabilityMajor { get; }
    public int ComputeCapabilityMinor { get; }
    public bool IsAvailable { get; }
}

// GPU utilization sample
public class GPUUtilizationSample
{
    public DateTime Timestamp { get; }
    public int DeviceId { get; }
    public float UtilizationPercent { get; } // 0.0 to 100.0
    public long UsedMemoryBytes { get; }
    public long FreeMemoryBytes { get; }
    public long TotalMemoryBytes { get; }
    public float TemperatureCelsius { get; }
    public float PowerUsageWatts { get; }
    public long FanSpeedRPM { get; }
}

// GPU statistics
public class GPUStatistics
{
    public int DeviceId { get; }

    // Utilization
    public float AverageUtilizationPercent { get; }
    public float MaxUtilizationPercent { get; }
    public float MinUtilizationPercent { get; }
    public TimeSpan TotalIdleTime { get; }

    // Memory
    public long PeakUsedMemoryBytes { get; }
    public float AverageMemoryUsagePercent { get; }
    public long TotalAllocations { get; }
    public long TotalDeallocations { get; }

    // Temperature and power
    public float AverageTemperatureCelsius { get; }
    public float MaxTemperatureCelsius { get; }
    public float AveragePowerUsageWatts { get; }
    public float PeakPowerUsageWatts { get; }

    // Samples
    public List<GPUUtilizationSample> Samples { get; }
}

// GPU tracker interface
public interface IGPUTracker
{
    // Device information
    List<GPUDeviceInfo> GetAvailableDevices();
    GPUDeviceInfo GetDeviceInfo(int deviceId);

    // Sampling
    void StartTracking(int deviceId = -1); // -1 = all devices
    void StopTracking();
    void SampleUtilization(); // Take immediate sample
    bool IsTracking { get; }

    // Statistics
    GPUStatistics GetStatistics(int deviceId);
    Dictionary<int, GPUStatistics> GetAllStatistics();

    // Timeline
    IEnumerable<GPUUtilizationSample> GetSamples(int deviceId, DateTime start, DateTime end);

    // Configuration
    void SetSamplingInterval(TimeSpan interval);
    void Enable();
    void Disable();
    bool IsEnabled { get; }
}

public class GPUUtilizationTracker : IGPUTracker
{
    public GPUUtilizationTracker(IStorageBackend storage);
    public GPUUtilizationTracker(IEventPublisher eventPublisher);

    // Configuration
    public int DefaultSamplingIntervalMs { get; set; } = 1000;
    public bool TrackTemperature { get; set; } = true;
    public bool TrackPower { get; set; } = true;
}
```

## Implementation Requirements

### 1. GPUDeviceInfo and GPUUtilizationSample (20-30 min)
- Implement `GPUDeviceInfo` with device properties
- Implement `GPUUtilizationSample` with metrics:
  - Utilization percentage
  - Memory usage
  - Temperature
  - Power usage
  - Fan speed
- Add validation for percentage values (0-100)
- Handle missing metrics (temperature, power may not be available on all devices)

### 2. GPUStatistics (30-45 min)
- Implement `GPUStatistics` with aggregated metrics:
  - Utilization: average, max, min, idle time
  - Memory: peak, average, allocations/deallocations
  - Temperature: average, max
  - Power: average, peak
- Compute statistics from samples:
  - Average of utilization percentages
  - Max/min values
  - Idle time (utilization < threshold)
- Store samples for timeline analysis

### 3. GPUUtilizationTracker Core (45-60 min)
- Implement `IGPUTracker` interface
- Query available GPU devices:
  - Use vendor-specific APIs (CUDA, ROCm, etc.)
  - Cache device information
- Start/stop tracking:
  - Create background task for sampling
  - Sample at configurable intervals
  - Stop when requested
- Sample GPU metrics:
  - Call vendor-specific APIs to get utilization
  - Get memory usage
  - Get temperature (if available)
  - Get power usage (if available)
- Store samples for analysis
- Integrate with event system (publish GPU utilization events)
- Integrate with storage backend

### 4. Vendor-Specific Implementations (45-60 min)
- Implement CUDA GPU tracking (if CUDA is available):
  - Use NVIDIA Management Library (NVML) if available
  - Fall back to CUDA runtime APIs
- Implement ROCm GPU tracking (if AMD GPUs are available):
  - Use ROCm SMI or ROCm profiler
- Implement generic tracking for other backends:
  - Provide stub implementations
  - Allow third-party plugins
- Use conditional compilation or runtime detection:
  - Only compile CUDA code if CUDA is available
  - Use reflection or dynamic loading for optional dependencies

### 5. Sampling and Analysis (30-45 min)
- Implement automatic sampling on timer:
  - Use `Timer` or `Task.Delay` for periodic sampling
  - Sample all tracked devices
- Implement idle time detection:
  - Define idle threshold (e.g., < 5% utilization)
  - Calculate total idle time
  - Report idle percentage
- Implement bottleneck detection:
  - Identify periods of high utilization with low throughput
  - Suggest optimization opportunities
- Provide timeline queries:
  - Get samples by time range
  - Get samples by step range

## File Structure
```
src/
  MLFramework.Visualization/
    GPU/
      GPUDeviceInfo.cs
      GPUUtilizationSample.cs
      GPUStatistics.cs
      IGPUTracker.cs
      GPUUtilizationTracker.cs
      Vendor/
        CudaGpuTracker.cs
        RocmGpuTracker.cs
        GenericGpuTracker.cs

tests/
  MLFramework.Visualization.Tests/
    GPU/
      GPUUtilizationTrackerTests.cs
      GPUStatisticsTests.cs
```

## Dependencies
- `MLFramework.Visualization.Events` (Event system)
- `MLFramework.Visualization.Storage` (Storage backend)
- Vendor-specific libraries (optional):
  - NVML (NVIDIA)
  - ROCm SMI (AMD)

## Integration Points
- Used by training loops to monitor GPU usage
- Integrated with profiler for performance analysis
- Data consumed by hardware visualization in TensorBoard

## Success Criteria
- Sampling overhead < 0.1% of training time
- Correctly identifies GPU devices and capabilities
- Accurate utilization tracking (validated against vendor tools)
- Sampling at 1Hz doesn't impact training performance
- Statistics correctly aggregate samples
- Gracefully handles missing metrics (temperature, power)
- Unit tests verify all functionality (with mocked GPU APIs)
