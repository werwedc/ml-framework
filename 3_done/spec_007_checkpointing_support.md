# Spec: Checkpointing Support

## Overview
Implement checkpointing functionality to save and load mixed-precision optimizer state, including master weights and loss scaler state.

## Dependencies
- Spec 002: MixedPrecisionOptions
- Spec 003: DynamicLossScaler
- Spec 006: MixedPrecisionOptimizer

## Implementation Details

### Checkpoint Data Structures
Create the checkpoint classes in `src/MLFramework/Optimizers/MixedPrecision/Checkpointing.cs`:

```csharp
using System;
using System.Collections.Generic;

namespace MLFramework.Optimizers.MixedPrecision;

/// <summary>
/// Checkpoint data for mixed-precision optimizer
/// </summary>
[Serializable]
public class MixedPrecisionCheckpoint
{
    /// <summary>
    /// Version of checkpoint format
    /// </summary>
    public int Version { get; set; } = 1;

    /// <summary>
    /// Timestamp when checkpoint was created
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Master weights (FP32) as dictionary of parameter name to byte array
    /// </summary>
    public Dictionary<string, byte[]> MasterWeights { get; set; }

    /// <summary>
    /// Training weights (target precision) as dictionary of parameter name to byte array
    /// </summary>
    public Dictionary<string, byte[]> TrainingWeights { get; set; }

    /// <summary>
    /// Current loss scale
    /// </summary>
    public float CurrentLossScale { get; set; }

    /// <summary>
    /// Steps since last overflow
    /// </summary>
    public int StepsSinceLastOverflow { get; set; }

    /// <summary>
    /// Consecutive overflows count
    /// </summary>
    public int ConsecutiveOverflows { get; set; }

    /// <summary>
    /// Total overflows count
    /// </summary>
    public int TotalOverflows { get; set; }

    /// <summary>
    /// Total step count
    /// </summary>
    public int StepCount { get; set; }

    /// <summary>
    /// Whether optimizer has fallen back to FP32
    /// </summary>
    public bool HasFallback { get; set; }

    /// <summary>
    /// Skipped steps count
    /// </summary>
    public int SkippedSteps { get; set; }

    /// <summary>
    /// Mixed precision options (serialized)
    /// </summary>
    public MixedPrecisionOptions Options { get; set; }

    /// <summary>
    /// Base optimizer state (type-specific)
    /// </summary>
    public object BaseOptimizerState { get; set; }

    /// <summary>
    /// Custom metadata for user information
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();
}

/// <summary>
/// Handles checkpoint save and load operations for mixed-precision optimizer
/// </summary>
public static class MixedPrecisionCheckpointManager
{
    #region Save Methods

    /// <summary>
    /// Creates a checkpoint from the optimizer state
    /// </summary>
    public static MixedPrecisionCheckpoint CreateCheckpoint(
        MixedPrecisionOptimizer optimizer,
        object baseOptimizerState = null,
        Dictionary<string, string> metadata = null)
    {
        if (optimizer == null)
            throw new ArgumentNullException(nameof(optimizer));

        var checkpoint = new MixedPrecisionCheckpoint
        {
            Timestamp = DateTime.UtcNow,
            MasterWeights = SerializeWeights(optimizer.MasterWeights),
            TrainingWeights = SerializeWeights(optimizer.TrainingWeights),
            Options = optimizer.Options.Clone(),
            HasFallback = optimizer.HasFallback,
            StepCount = optimizer.StepCount,
            SkippedSteps = optimizer.SkippedSteps,
            BaseOptimizerState = baseOptimizerState
        };

        // Extract loss scaler state
        var stats = optimizer.GetStats().LossScalerStats;
        checkpoint.CurrentLossScale = stats.CurrentScale;
        checkpoint.StepsSinceLastOverflow = stats.StepsSinceLastOverflow;
        checkpoint.ConsecutiveOverflows = stats.ConsecutiveOverflows;
        checkpoint.TotalOverflows = stats.TotalOverflows;

        // Add metadata
        if (metadata != null)
        {
            foreach (var kvp in metadata)
            {
                checkpoint.Metadata[kvp.Key] = kvp.Value;
            }
        }

        // Add system metadata
        checkpoint.Metadata["precision"] = optimizer.TargetPrecision.ToString();
        checkpoint.Metadata["version"] = "1.0";

        return checkpoint;
    }

    /// <summary>
    /// Saves checkpoint to a file
    /// </summary>
    public static void SaveCheckpoint(
        MixedPrecisionCheckpoint checkpoint,
        string filepath)
    {
        if (checkpoint == null)
            throw new ArgumentNullException(nameof(checkpoint));

        if (string.IsNullOrWhiteSpace(filepath))
            throw new ArgumentException("Filepath cannot be null or empty", nameof(filepath));

        // TODO: Implement actual serialization (e.g., JSON, binary, etc.)
        // For now, this is a placeholder
        throw new NotImplementedException("Checkpoint serialization not yet implemented");
    }

    #endregion

    #region Load Methods

    /// <summary>
    /// Loads checkpoint from a file
    /// </summary>
    public static MixedPrecisionCheckpoint LoadCheckpoint(string filepath)
    {
        if (string.IsNullOrWhiteSpace(filepath))
            throw new ArgumentException("Filepath cannot be null or empty", nameof(filepath));

        // TODO: Implement actual deserialization
        // For now, this is a placeholder
        throw new NotImplementedException("Checkpoint deserialization not yet implemented");
    }

    /// <summary>
    /// Restores optimizer state from checkpoint
    /// </summary>
    public static void RestoreCheckpoint(
        MixedPrecisionOptimizer optimizer,
        MixedPrecisionCheckpoint checkpoint,
        bool restoreWeights = true,
        bool restoreState = true)
    {
        if (optimizer == null)
            throw new ArgumentNullException(nameof(optimizer));

        if (checkpoint == null)
            throw new ArgumentNullException(nameof(checkpoint));

        if (restoreWeights)
        {
            // Restore master weights
            var masterWeights = DeserializeWeights(checkpoint.MasterWeights);
            var trainingWeights = DeserializeWeights(checkpoint.TrainingWeights);

            // Set parameters on optimizer
            optimizer.SetParameters(masterWeights);
        }

        if (restoreState)
        {
            // Restore loss scaler state
            var lossScalerField = optimizer.GetType()
                .GetField("_lossScaler", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            if (lossScalerField != null)
            {
                var lossScaler = lossScalerField.GetValue(optimizer) as DynamicLossScaler;
                if (lossScaler != null)
                {
                    // TODO: Add RestoreState method to DynamicLossScaler
                    // lossScaler.RestoreState(checkpoint);
                }
            }
        }
    }

    /// <summary>
    /// Convenience method: Load checkpoint and restore optimizer in one call
    /// </summary>
    public static void LoadAndRestore(
        MixedPrecisionOptimizer optimizer,
        string filepath,
        bool restoreWeights = true,
        bool restoreState = true)
    {
        var checkpoint = LoadCheckpoint(filepath);
        RestoreCheckpoint(optimizer, checkpoint, restoreWeights, restoreState);
    }

    #endregion

    #region Utility Methods

    /// <summary>
    /// Serializes weights dictionary to byte arrays
    /// </summary>
    private static Dictionary<string, byte[]> SerializeWeights(
        IReadOnlyDictionary<string, ITensor> weights)
    {
        var serialized = new Dictionary<string, byte[]>();

        foreach (var kvp in weights)
        {
            // TODO: Implement actual tensor serialization
            // For now, placeholder
            serialized[kvp.Key] = Array.Empty<byte>();
        }

        return serialized;
    }

    /// <summary>
    /// Deserializes weights from byte arrays
    /// </summary>
    private static Dictionary<string, ITensor> DeserializeWeights(
        Dictionary<string, byte[]> serialized)
    {
        var weights = new Dictionary<string, ITensor>();

        foreach (var kvp in serialized)
        {
            // TODO: Implement actual tensor deserialization
            // For now, placeholder
            weights[kvp.Key] = null;
        }

        return weights;
    }

    /// <summary>
    /// Validates checkpoint integrity
    /// </summary>
    public static bool ValidateCheckpoint(MixedPrecisionCheckpoint checkpoint)
    {
        if (checkpoint == null)
            return false;

        if (checkpoint.Version <= 0)
            return false;

        if (checkpoint.MasterWeights == null || checkpoint.MasterWeights.Count == 0)
            return false;

        if (checkpoint.TrainingWeights == null || checkpoint.TrainingWeights.Count == 0)
            return false;

        if (checkpoint.Options == null)
            return false;

        return true;
    }

    #endregion
}
```

## Requirements

### Functional Requirements
1. **Checkpoint Creation**: Capture all optimizer state in a checkpoint object
2. **Save to File**: Serialize checkpoint to disk
3. **Load from File**: Deserialize checkpoint from disk
4. **Restore State**: Apply checkpoint to optimizer
5. **Weight Serialization**: Save master and training weights
6. **State Preservation**: Save loss scaler state and step counts
7. **Metadata Support**: Allow custom metadata in checkpoints
8. **Validation**: Verify checkpoint integrity

### Non-Functional Requirements
1. **Backward Compatibility**: Include version number for future changes
2. **Efficiency**: Serialization should be fast and space-efficient
3. **Portability**: Checkpoints should be machine-independent (if possible)
4. **Error Handling**: Handle corrupted or invalid checkpoints gracefully

## Checkpoint Contents

### Required Fields
- Master weights (FP32) - critical for resumption
- Training weights - optional but useful
- Loss scale - critical for stability
- Overflow counters - important for dynamic scaling
- Step count - for training resumption
- Fallback state - maintain mode

### Optional Fields
- Metadata - user-defined information
- Base optimizer state - type-specific state
- Performance stats - monitoring information

## Deliverables

### Source Files
1. `src/MLFramework/Optimizers/MixedPrecision/Checkpointing.cs`

### Unit Tests
- Tests will be covered in spec 011 (MixedPrecisionOptimizer unit tests)

## Notes for Coder
- Serialization/deserialization should be stubbed for now (tensor API not ready)
- Focus on the checkpoint structure and API design
- Consider using JSON for human-readable checkpoints (future enhancement)
- Binary format may be better for performance (future enhancement)
- Versioning is critical for backward compatibility
- Add RestoreState method to DynamicLossScaler in a follow-up spec
- Metadata is useful for storing hyperparameters, dataset info, etc.
