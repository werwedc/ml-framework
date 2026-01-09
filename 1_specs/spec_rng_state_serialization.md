# Spec: RNG State Serialization

## Overview
Implement RNG state capture and restore functionality to enable exact reproducibility through checkpointing. This spec adds the `RNGSnapshot` class and extends `SeedManager` with state management methods.

## Technical Requirements

### RNGSnapshot Class Definition
```csharp
namespace MLFramework.Utilities;

/// <summary>
/// Serializable snapshot of all RNG states in the system
/// </summary>
[Serializable]
public class RNGSnapshot
{
    /// <summary>
    /// Timestamp when snapshot was created
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// CPU random generator state
    /// </summary>
    public int RandomSeed { get; set; }

    /// <summary>
    /// NumPy RNG state (serialized)
    /// </summary>
    public byte[]? NumpyState { get; set; }

    /// <summary>
    /// CUDA RNG state per device
    /// </summary>
    public Dictionary<int, byte[]> CudaStates { get; set; }

    /// <summary>
    /// Additional metadata for debugging
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; }

    public RNGSnapshot()
    {
        Timestamp = DateTime.UtcNow;
        CudaStates = new Dictionary<int, byte[]>();
        Metadata = new Dictionary<string, object>();
    }
}
```

### Extended SeedManager Methods
```csharp
public class SeedManager : IDisposable
{
    // ... existing methods ...

    /// <summary>
    /// Captures the current state of all RNGs
    /// </summary>
    /// <returns>RNGSnapshot containing current RNG states</returns>
    public RNGSnapshot CaptureRNGState();

    /// <summary>
    /// Restores RNGs to a previously captured state
    /// </summary>
    /// <param name="snapshot">The snapshot to restore</param>
    public void RestoreRNGState(RNGSnapshot snapshot);

    /// <summary>
    /// Saves an RNG snapshot to a file
    /// </summary>
    /// <param name="snapshot">The snapshot to save</param>
    /// <param name="filePath">Path to save the snapshot</param>
    public void SaveRNGSnapshot(RNGSnapshot snapshot, string filePath);

    /// <summary>
    /// Loads an RNG snapshot from a file
    /// </summary>
    /// <param name="filePath">Path to load the snapshot from</param>
    /// <returns>The loaded RNGSnapshot</returns>
    public RNGSnapshot LoadRNGSnapshot(string filePath);
}
```

### Implementation Details

#### CaptureRNGState Method
- Create new RNGSnapshot instance
- Capture current CPU random seed from stored value
- Capture NumPy state (placeholder for interop)
- Capture CUDA state per device (placeholder for CUDA integration)
- Add metadata: timestamp, seed manager version, etc.
- Return the snapshot

#### RestoreRNGState Method
- Validate snapshot is not null
- Restore CPU random seed by calling SetRandomSeed
- Restore NumPy state (placeholder)
- Restore CUDA states per device (placeholder)
- Update _currentSeed to match snapshot's RandomSeed

#### SaveRNGSnapshot Method
- Use System.Text.Json or BinaryFormatter for serialization
- Ensure file directory exists
- Handle serialization exceptions
- Validate snapshot before saving

#### LoadRNGSnapshot Method
- Validate file exists
- Deserialize file to RNGSnapshot
- Validate loaded snapshot
- Return the snapshot or throw on error

### Design Decisions

1. **Serialization Format**: Use JSON for human-readable debugging, but consider binary for performance
2. **Per-Device CUDA State**: Store CUDA state per device ID to support multi-GPU
3. **Metadata Dictionary**: Allow extensibility for future RNG types
4. **Validation**: Validate snapshots before restoration to prevent corruption

### Directory Structure
```
src/
  MLFramework.Utilities/
    SeedManager.cs (extended)
    RNGSnapshot.cs
```

## Success Criteria
1. ✅ RNGSnapshot can be instantiated and serialized
2. ✅ CaptureRNGState creates a valid snapshot
3. ✅ RestoreRNGState restores seeds correctly
4. ✅ SaveRNGSnapshot writes snapshot to file
5. ✅ LoadRNGSnapshot reads and deserializes snapshot
6. ✅ Snapshot includes timestamp and metadata

## Dependencies
- spec_seed_manager_core.md - Base SeedManager class
- System.Text.Json for JSON serialization (or similar)

## Notes
- NumPy and CUDA state capture are placeholders for backend integration
- Consider versioning snapshots for backward compatibility
- Add error handling for corrupted snapshot files

## Related Specs
- spec_seed_manager_core.md - Base class that this extends
- spec_multidevice_seeding.md - Will use per-device state capture
