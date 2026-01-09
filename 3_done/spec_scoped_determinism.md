# Spec: Scoped Determinism

## Overview
Implement scoped determinism using C#'s `IDisposable` pattern to enable temporary deterministic behavior within a specific scope. This allows fine-grained control over reproducibility in critical sections while maintaining performance elsewhere.

## Technical Requirements

### ScopedContext Class
```csharp
namespace MLFramework.Utilities;

/// <summary>
/// Context manager for scoped determinism behavior
/// </summary>
public sealed class ScopedContext : IDisposable
{
    private readonly SeedManager _seedManager;
    private readonly DeterministicModeFlags _previousMode;
    private readonly RNGSnapshot? _previousRngState;
    private readonly int? _previousSeed;
    private readonly bool _restoreState;
    private bool _disposed;

    /// <summary>
    /// Creates a new scoped context
    /// </summary>
    /// <param name="seedManager">The seed manager instance</param>
    /// <param name="previousMode">The mode to restore on dispose</param>
    /// <param name="previousRngState">The RNG state to restore (optional)</param>
    /// <param name="previousSeed">The seed to restore (optional)</param>
    /// <param name="restoreState">Whether to restore RNG state on exit</param>
    internal ScopedContext(
        SeedManager seedManager,
        DeterministicModeFlags previousMode,
        RNGSnapshot? previousRngState = null,
        int? previousSeed = null,
        bool restoreState = false)
    {
        _seedManager = seedManager;
        _previousMode = previousMode;
        _previousRngState = previousRngState;
        _previousSeed = previousSeed;
        _restoreState = restoreState;
        _disposed = false;
    }

    /// <summary>
    /// Restores the previous state when exiting the scope
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;

        // Restore deterministic mode
        _seedManager.SetDeterministicMode(_previousMode);

        // Restore RNG state if requested
        if (_restoreState && _previousRngState != null)
        {
            _seedManager.RestoreRNGState(_previousRngState);
        }
        // Otherwise restore seed
        else if (_previousSeed.HasValue)
        {
            _seedManager.SetGlobalSeed(_previousSeed.Value);
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
```

### Extended SeedManager Methods
```csharp
public class SeedManager : IDisposable
{
    // ... existing methods ...

    /// <summary>
    /// Creates a scoped context with deterministic behavior enabled
    /// </summary>
    /// <param name="enabled">Whether to enable determinism in the scope</param>
    /// <returns>ScopedContext that restores previous state on disposal</returns>
    public IDisposable WithDeterminism(bool enabled);

    /// <summary>
    /// Creates a scoped context with a specific global seed
    /// </summary>
    /// <param name="seed">The seed to use within the scope</param>
    /// <returns>ScopedContext that restores previous state on disposal</returns>
    public IDisposable WithGlobalSeed(int seed);

    /// <summary>
    /// Creates a scoped context with specific deterministic mode flags
    /// </summary>
    /// <param name="flags">The deterministic mode flags to use within the scope</param>
    /// <returns>ScopedContext that restores previous state on disposal</returns>
    public IDisposable WithDeterministicMode(DeterministicModeFlags flags);

    /// <summary>
    /// Creates a scoped context that restores RNG state on exit
    /// </summary>
    /// <param name="enabled">Whether to enable determinism in the scope</param>
    /// <param name="restoreState">Whether to capture and restore full RNG state</param>
    /// <returns>ScopedContext that restores previous state on disposal</returns>
    public IDisposable WithDeterminism(bool enabled, bool restoreState);
}
```

### Implementation Details

#### WithDeterminism Method
- Capture current deterministic mode from IsDeterministic
- Create ScopedContext with current mode as previous mode
- If enabled: SetDeterministicMode(DeterministicModeFlags.All)
- If disabled: SetDeterministicMode(DeterministicModeFlags.None)
- Return the ScopedContext

#### WithGlobalSeed Method
- Capture current seed from CurrentSeed
- Capture current RNG state
- Create ScopedContext with previous seed and RNG state
- Set new global seed
- Return the ScopedContext with restoreState=true

#### WithDeterministicMode Method
- Capture current deterministic mode from IsDeterministic
- Create ScopedContext with current mode as previous mode
- Set new deterministic mode using SetDeterministicMode(flags)
- Return the ScopedContext

#### WithDeterminism Method (Overload)
- Capture current deterministic mode from IsDeterministic
- Capture current RNG state if restoreState is true
- Create ScopedContext with previous mode, RNG state, and restoreState flag
- Set new deterministic mode
- Return the ScopedContext

#### ScopedContext.Dispose Method
- Restore previous deterministic mode
- Restore RNG state if restoreState was true
- Restore seed otherwise
- Mark as disposed to prevent double-dispose

### Design Decisions

1. **IDisposable Pattern**: Use C#'s standard disposal pattern for scope management
2. **State Restoration**: Automatically restore previous state on scope exit
3. **Optional State Restoration**: Allow control over whether to restore full RNG state
4. **Thread Safety**: Each ScopedContext is independent and thread-safe

### Usage Examples

#### Basic Scoped Determinism
```csharp
using var manager = new SeedManager();

// Enable determinism for critical section
using (manager.WithDeterminism(true))
{
    // Code here is deterministic
    var results = model.Train(data);
}

// Determinism automatically disabled here
```

#### Scoped Seeding
```csharp
using var manager = new SeedManager();

// Use specific seed for reproducible section
using (manager.WithGlobalSeed(42))
{
    var results1 = model.Predict(data1);
}

// Different seed
using (manager.WithGlobalSeed(43))
{
    var results2 = model.Predict(data2);
}
```

#### Scoped Deterministic Mode
```csharp
using var manager = new SeedManager();

// Enable only cuDNN determinism
using (manager.WithDeterministicMode(DeterministicModeFlags.CudnnDeterministic))
{
    // Only cuDNN is deterministic, cuBLAS is not
    var results = model.Train(data);
}
```

### Directory Structure
```
src/
  MLFramework.Utilities/
    SeedManager.cs (extended)
    ScopedContext.cs
```

## Success Criteria
1. ✅ ScopedContext properly restores deterministic mode on dispose
2. ✅ ScopedContext properly restores RNG state on dispose when requested
3. ✅ WithDeterminism enables/disables determinism within scope
4. ✅ WithGlobalSeed sets seed within scope and restores on exit
5. ✅ WithDeterministicMode sets specific flags within scope
6. ✅ Nested scopes work correctly (inner scope restores to outer scope's state)
7. ✅ Double-dispose is handled gracefully

## Dependencies
- spec_seed_manager_core.md - Base SeedManager class
- spec_rng_state_serialization.md - RNGSnapshot class
- spec_deterministic_mode.md - DeterministicModeFlags

## Notes
- Ensure thread safety when using scoped contexts in parallel code
- Consider adding WithAsyncDeterminism for async scenarios
- Document that nested scopes restore to immediate outer scope's state

## Related Specs
- spec_seed_manager_core.md - Base class that this extends
- spec_rng_state_serialization.md - Provides state capture/restore
- spec_deterministic_mode.md - Provides deterministic mode control
