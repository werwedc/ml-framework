# Spec: SeedManager Core Class

## Overview
Implement the core `SeedManager` class that provides centralized control over random number generators (RNGs) used throughout the framework. This spec focuses on basic seeding functionality for CPU, NumPy, and CUDA RNGs.

## Technical Requirements

### Class Definition
```csharp
namespace MLFramework.Utilities;

public class SeedManager : IDisposable
{
    private int _currentSeed;
    private readonly object _lock = new();

    /// <summary>
    /// Sets a global seed for all RNGs (random, NumPy, CUDA)
    /// </summary>
    /// <param name="seed">The seed value to use</param>
    public void SetGlobalSeed(int seed);

    /// <summary>
    /// Seeds the CPU random number generator
    /// </summary>
    /// <param name="seed">The seed value to use</param>
    public void SetRandomSeed(int seed);

    /// <summary>
    /// Seeds the NumPy random number generator
    /// </summary>
    /// <param name="seed">The seed value to use</param>
    public void SetNumpySeed(int seed);

    /// <summary>
    /// Seeds the CUDA random number generator
    /// </summary>
    /// <param name="seed">The seed value to use</param>
    public void SetCudaSeed(int seed);

    /// <summary>
    /// Gets the current global seed value
    /// </summary>
    public int CurrentSeed => _currentSeed;

    /// <summary>
    /// Disposes the seed manager and cleans up resources
    /// </summary>
    public void Dispose();
}
```

### Implementation Details

#### SetGlobalSeed Method
- Store the seed value in `_currentSeed`
- Call `SetRandomSeed(seed)`, `SetNumpySeed(seed)`, and `SetCudaSeed(seed)`
- Thread-safe using `_lock`

#### SetRandomSeed Method
- For C# `System.Random`: Store seed for new Random instances
- Document that existing Random instances are not affected
- Consider using a thread-safe Random pool

#### SetNumpySeed Method
- Note: This is a placeholder for interop with NumPy (if using Python interop)
- For pure C# implementation, this may be a no-op or use equivalent library
- Design interface to be compatible with future NumPy integration

#### SetCudaSeed Method
- Interface for CUDA RNG seeding
- Placeholder for CUDA-specific implementation
- Should handle cases where CUDA is not available

### Design Decisions

1. **Thread Safety**: Use lock to ensure thread-safe seed operations
2. **Immutability of Existing RNGs**: Existing Random instances are not reseeded to avoid unexpected behavior
3. **Separate Seed Methods**: Allow individual component seeding for fine-grained control

### Directory Structure
```
src/
  MLFramework.Utilities/
    SeedManager.cs
```

## Success Criteria
1. ✅ SeedManager can be instantiated and disposed
2. ✅ SetGlobalSeed seeds all RNG components
3. ✅ Individual seed methods work independently
4. ✅ CurrentSeed property returns the last set seed
5. ✅ Thread-safe seed operations

## Dependencies
- None (self-contained core class)

## Notes
- This is the foundation class that will be extended in subsequent specs
- Methods for NumPy and CUDA are interfaces for future backend integration
- Consider adding logging for debugging seed operations

## Related Specs
- spec_rng_state_serialization.md - Adds state capture/restore
- spec_deterministic_mode.md - Adds deterministic mode configuration
- spec_scoped_determinism.md - Adds scoped behavior support
