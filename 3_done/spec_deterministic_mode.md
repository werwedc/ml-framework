# Spec: Deterministic Mode Configuration

## Overview
Implement deterministic mode configuration that controls backend-specific settings for reproducible computations. This spec adds flags and configuration methods to enable/disable deterministic algorithms in cuDNN, cuBLAS, and other backends.

## Technical Requirements

### DeterministicModeFlags Enum
```csharp
namespace MLFramework.Utilities;

[Flags]
public enum DeterministicModeFlags
{
    /// <summary>
    /// No deterministic mode enabled
    /// </summary>
    None = 0,

    /// <summary>
    /// Enable deterministic algorithms in cuDNN
    /// </summary>
    CudnnDeterministic = 1 << 0,

    /// <summary>
    /// Enable deterministic algorithms in cuBLAS
    /// </summary>
    CublasDeterministic = 1 << 1,

    /// <summary>
    /// Enable deterministic memory allocation in CUDA
    /// </summary>
    CudaMemoryDeterministic = 1 << 2,

    /// <summary>
    /// Enable CUDA Graphs for deterministic kernel launch ordering
    /// </summary>
    CudaGraphs = 1 << 3,

    /// <summary>
    /// Enable all deterministic modes
    /// </summary>
    All = CudnnDeterministic | CublasDeterministic | CudaMemoryDeterministic | CudaGraphs
}
```

### Extended SeedManager Methods
```csharp
public class SeedManager : IDisposable
{
    // ... existing methods ...

    private DeterministicModeFlags _deterministicMode = DeterministicModeFlags.None;

    /// <summary>
    /// Gets or sets the current deterministic mode flags
    /// </summary>
    public DeterministicModeFlags IsDeterministic
    {
        get => _deterministicMode;
        set => SetDeterministicMode(value);
    }

    /// <summary>
    /// Enables or disables specific deterministic modes
    /// </summary>
    /// <param name="flags">Deterministic mode flags to set</param>
    public void SetDeterministicMode(DeterministicModeFlags flags);

    /// <summary>
    /// Enables a specific deterministic mode flag
    /// </summary>
    /// <param name="flag">The flag to enable</param>
    public void EnableDeterministicMode(DeterministicModeFlags flag);

    /// <summary>
    /// Disables a specific deterministic mode flag
    /// </summary>
    /// <param name="flag">The flag to disable</param>
    public void DisableDeterministicMode(DeterministicModeFlags flag);

    /// <summary>
    /// Checks if a specific deterministic mode is enabled
    /// </summary>
    /// <param name="flag">The flag to check</param>
    /// <returns>True if the flag is enabled</returns>
    public bool IsDeterministicModeEnabled(DeterministicModeFlags flag);

    /// <summary>
    /// Gets a description of performance impact for current deterministic settings
    /// </summary>
    public string GetPerformanceImpact();
}
```

### Implementation Details

#### SetDeterministicMode Method
- Store the flags in _deterministicMode
- Call backend-specific configuration methods:
  - If CudnnDeterministic: Set CUDNN_DETERMINISTIC=1 environment variable
  - If CublasDeterministic: Configure cuBLAS to use PEDANTIC mode
  - If CudaMemoryDeterministic: Set cudaDeviceSetLimit for memory pool
  - If CudaGraphs: Enable CUDA Graph capture mode
- Log mode changes for debugging

#### EnableDeterministicMode Method
- Add the flag to _deterministicMode using bitwise OR
- Call SetDeterministicMode with updated flags

#### DisableDeterministicMode Method
- Remove the flag from _deterministicMode using bitwise AND with complement
- Call SetDeterministicMode with updated flags

#### IsDeterministicModeEnabled Method
- Check if flag is set in _deterministicMode using bitwise AND
- Return true if flag is set, false otherwise

#### GetPerformanceImpact Method
- Return estimated performance degradation based on enabled flags
- Example: "CudnnDeterministic: ~20% slower, CublasDeterministic: ~15% slower"
- Provide guidance on when to enable/disable specific flags

### Design Decisions

1. **Flags Pattern**: Use [Flags] enum to allow multiple deterministic modes simultaneously
2. **Environment Variables**: Use environment variables for backend configuration (cuDNN, cuBLAS)
3. **Performance Impact**: Provide transparency about performance trade-offs
4. **Backend Isolation**: Each backend is controlled independently

### Backend Configuration Details

#### cuDNN Configuration
- Set environment variable: `CUDNN_DETERMINISTIC=1`
- This forces cuDNN to use deterministic convolution algorithms
- May impact convolution performance by 10-30%

#### cuBLAS Configuration
- Configure cuBLAS workspace to use deterministic algorithms
- Set math mode to `CUBLAS_MATH_MODE_PEDANTIC`
- May impact GEMM operations by 15-25%

#### CUDA Memory Configuration
- Set memory pool size limit for deterministic allocation
- Use `cudaDeviceSetLimit(cudaLimitMemPoolSize)`
- May reduce memory fragmentation

#### CUDA Graphs Configuration
- Enable graph capture mode
- Ensures deterministic kernel launch ordering
- May improve performance for repeated operations

### Directory Structure
```
src/
  MLFramework.Utilities/
    SeedManager.cs (extended)
    DeterministicModeFlags.cs
```

## Success Criteria
1. ✅ DeterministicModeFlags enum is correctly defined with all flags
2. ✅ SetDeterministicMode configures backend settings correctly
3. ✅ Enable/Disable methods work with individual flags
4. ✅ IsDeterministicModeEnabled correctly checks flag state
5. ✅ GetPerformanceImpact provides meaningful performance estimates
6. ✅ Environment variables are set for cuDNN/cuBLAS when appropriate

## Dependencies
- spec_seed_manager_core.md - Base SeedManager class
- Future: CUDA/cuDNN/cuBLAS interop libraries

## Notes
- Backend configuration methods are placeholders for actual CUDA interop
- Environment variables are the primary mechanism for now
- Consider adding platform-specific configuration in future iterations

## Related Specs
- spec_seed_manager_core.md - Base class that this extends
- spec_scoped_determinism.md - Will use deterministic mode for scoped behavior
- spec_validation_diagnostics.md - Will validate deterministic mode usage
