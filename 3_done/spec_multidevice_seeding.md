# Spec: Multi-Device Seeding

## Overview
Extend SeedManager to support deterministic seeding across multiple devices (GPUs) and data loading workers. This spec ensures reproducibility in distributed and parallel computing scenarios.

## Technical Requirements

### Extended SeedManager Methods
```csharp
public class SeedManager : IDisposable
{
    // ... existing methods ...

    private readonly Dictionary<int, int> _deviceSeeds = new();
    private readonly Dictionary<int, int> _workerSeeds = new();

    /// <summary>
    /// Seeds all available CUDA devices deterministically
    /// Uses formula: baseSeed + device_id
    /// </summary>
    /// <param name="baseSeed">The base seed value</param>
    /// <param name="deviceCount">Number of devices to seed (default: auto-detect)</param>
    public void SeedAllDevices(int baseSeed, int? deviceCount = null);

    /// <summary>
    /// Seeds a specific CUDA device
    /// </summary>
    /// <param name="deviceId">The device ID (0-based)</param>
    /// <param name="seed">The seed value for this device</param>
    public void SeedDevice(int deviceId, int seed);

    /// <summary>
    /// Gets the seed for a specific device
    /// </summary>
    /// <param name="deviceId">The device ID</param>
    /// <returns>The seed value for the device</returns>
    public int GetDeviceSeed(int deviceId);

    /// <summary>
    /// Seeds data loading workers deterministically
    /// Uses formula: baseSeed + worker_id
    /// </summary>
    /// <param name="baseSeed">The base seed value</param>
    /// <param name="workerCount">Number of workers to seed</param>
    public void SeedWorkers(int baseSeed, int workerCount);

    /// <summary>
    /// Seeds a specific data loading worker
    /// </summary>
    /// <param name="workerId">The worker ID (0-based)</param>
    /// <param name="seed">The seed value for this worker</param>
    public void SeedWorker(int workerId, int seed);

    /// <summary>
    /// Gets the seed for a specific worker
    /// </summary>
    /// <param name="workerId">The worker ID</param>
    /// <returns>The seed value for the worker</returns>
    public int GetWorkerSeed(int workerId);

    /// <summary>
    /// Gets a deterministic seed for a given device and operation
    /// Formula: baseSeed + device_id * 1000 + operation_id
    /// </summary>
    /// <param name="baseSeed">The base seed</param>
    /// <param name="deviceId">The device ID</param>
    /// <param name="operationId">The operation ID</param>
    /// <returns>A deterministic seed for this combination</returns>
    public int GetDeterministicSeed(int baseSeed, int deviceId, int operationId = 0);
}
```

### Implementation Details

#### SeedAllDevices Method
- Validate deviceCount (use default if null)
- For each device from 0 to deviceCount-1:
  - Calculate seed: baseSeed + deviceId
  - Call SeedDevice(deviceId, calculatedSeed)
  - Store seed in _deviceSeeds dictionary
- Log seeding operations for debugging

#### SeedDevice Method
- Validate deviceId >= 0
- Store seed in _deviceSeeds[deviceId]
- Call SetCudaSeed(seed) with device context (placeholder for CUDA interop)
- Update RNGSnapshot's CudaStates dictionary

#### GetDeviceSeed Method
- Validate deviceId exists in _deviceSeeds
- Return the stored seed
- Throw if device not seeded

#### SeedWorkers Method
- Validate workerCount > 0
- For each worker from 0 to workerCount-1:
  - Calculate seed: baseSeed + workerId
  - Call SeedWorker(workerId, calculatedSeed)
  - Store seed in _workerSeeds dictionary
- Log seeding operations for debugging

#### SeedWorker Method
- Validate workerId >= 0
- Store seed in _workerSeeds[workerId]
- Note: Worker seeding is tracked but actual RNG is managed by worker processes
- This provides a deterministic seed that workers can use

#### GetWorkerSeed Method
- Validate workerId exists in _workerSeeds
- Return the stored seed
- Throw if worker not seeded

#### GetDeterministicSeed Method
- Calculate deterministic seed using formula
- Formula: baseSeed + (deviceId * 1000) + operationId
- Ensure uniqueness across devices and operations
- Return the calculated seed

### Design Decisions

1. **Device-Specific Seeds**: Each device gets a unique seed (baseSeed + device_id)
2. **Worker-Specific Seeds**: Each worker gets a unique seed (baseSeed + worker_id)
3. **Deterministic Formula**: Use simple arithmetic formulas for reproducible seed generation
4. **State Tracking**: Track device and worker seeds in dictionaries for validation
5. **Operation-Level Seeds**: Provide method for operation-specific deterministic seeds

### Multi-GPU Scenarios

#### Single GPU Training
```csharp
var manager = new SeedManager();
manager.SetGlobalSeed(42);  // Seeds CPU, NumPy, and GPU 0
```

#### Multi-GPU Training (Data Parallel)
```csharp
var manager = new SeedManager();
manager.SeedAllDevices(42, deviceCount: 4);
// GPU 0: seed=42
// GPU 1: seed=43
// GPU 2: seed=44
// GPU 3: seed=45
```

#### Multi-GPU Training (Model Parallel)
```csharp
var manager = new SeedManager();
manager.SeedAllDevices(42, deviceCount: 2);
// Each device handles part of model with deterministic seed
```

### Data Loading Scenarios

#### Single Worker
```csharp
var manager = new SeedManager();
manager.SeedWorkers(100, workerCount: 1);
// Worker 0: seed=100
```

#### Multiple Workers
```csharp
var manager = new SeedManager();
manager.SeedWorkers(100, workerCount: 4);
// Worker 0: seed=100
// Worker 1: seed=101
// Worker 2: seed=102
// Worker 3: seed=103
```

### Deterministic Seeding Strategy

#### Formula Rationale
- **Base seed**: Provides reproducibility across runs
- **Device offset (device_id)**: Ensures different devices don't produce identical random numbers
- **Worker offset (worker_id)**: Ensures different workers don't produce identical random numbers
- **Operation offset (operation_id)**: Ensures different operations in same pipeline don't produce identical random numbers

#### Seed Collision Avoidance
- Use large multipliers (1000) to avoid collisions
- Document formula clearly for users
- Consider providing custom seed strategies for advanced use cases

### Directory Structure
```
src/
  MLFramework.Utilities/
    SeedManager.cs (extended)
```

## Success Criteria
1. ✅ SeedAllDevices seeds all specified devices with unique seeds
2. ✅ SeedDevice stores and tracks device-specific seeds
3. ✅ GetDeviceSeed returns correct seed for device
4. ✅ SeedWorkers seeds all workers with unique seeds
5. ✅ SeedWorker stores and tracks worker-specific seeds
6. ✅ GetWorkerSeed returns correct seed for worker
7. ✅ GetDeterministicSeed produces deterministic results across calls
8. ✅ Seeding is reproducible across runs with same base seed

## Dependencies
- spec_seed_manager_core.md - Base SeedManager class
- spec_rng_state_serialization.md - RNGSnapshot class for per-device states

## Notes
- CUDA-specific implementation is a placeholder for backend integration
- Worker seeding provides deterministic seeds but doesn't manage worker RNGs directly
- Consider adding distributed training support (multi-machine) in future iterations

## Related Specs
- spec_seed_manager_core.md - Base class that this extends
- spec_rng_state_serialization.md - Extends RNGSnapshot for per-device states
