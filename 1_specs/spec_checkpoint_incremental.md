# Spec: Incremental Checkpointing

## Overview
Implement incremental checkpointing that saves only changed parameters between checkpoints, reducing storage costs and I/O time for large models with sparse updates.

## Scope
- 30-45 minutes coding time
- Focus on delta detection and compression
- Target: `src/MLFramework/Checkpointing/Incremental/`

## Classes

### 1. IncrementalCheckpointManager (Main Manager)
```csharp
public class IncrementalCheckpointManager
{
    private readonly DistributedCheckpoint _checkpoint;
    private readonly IChecksumCalculator _checksumCalculator;
    private readonly ICompressionProvider _compressionProvider;
    private readonly ILogger<IncrementalCheckpointManager> _logger;
    private readonly Dictionary<string, IncrementalSnapshot> _snapshots = new();

    public IncrementalCheckpointManager(
        DistributedCheckpoint checkpoint,
        IChecksumCalculator? checksumCalculator = null,
        ICompressionProvider? compressionProvider = null,
        ILogger<IncrementalCheckpointManager>? logger = null)
    {
        _checkpoint = checkpoint;
        _checksumCalculator = checksumCalculator ?? new SHA256ChecksumCalculator();
        _compressionProvider = compressionProvider ?? new ZstdCompressionProvider();
        _logger = logger;
    }

    /// <summary>
    /// Save a full checkpoint and create a baseline snapshot
    /// </summary>
    public async Task<string> SaveBaselineAsync(
        IStateful model,
        IStateful optimizer,
        SaveOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new SaveOptions();
        var checkpointId = options.CheckpointPrefix ?? GenerateCheckpointId("baseline");

        _logger?.LogInformation("Saving baseline checkpoint: {CheckpointId}", checkpointId);

        // Save full checkpoint
        var checkpointPath = await _checkpoint.SaveAsync(model, optimizer, options, cancellationToken);

        // Create snapshot
        var snapshot = await CreateSnapshotAsync(model, optimizer, cancellationToken);
        _snapshots[checkpointId] = snapshot;

        _logger?.LogInformation("Baseline checkpoint saved: {CheckpointId}", checkpointId);
        return checkpointId;
    }

    /// <summary>
    /// Save an incremental checkpoint (only changed parameters)
    /// </summary>
    public async Task<string> SaveIncrementalAsync(
        IStateful model,
        IStateful optimizer,
        string baselineCheckpointId,
        SaveOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new SaveOptions();
        var checkpointId = options.CheckpointPrefix ?? GenerateCheckpointId("incremental");

        _logger?.LogInformation("Saving incremental checkpoint: {CheckpointId}", checkpointId);

        // Get baseline snapshot
        if (!_snapshots.TryGetValue(baselineCheckpointId, out var baselineSnapshot))
        {
            throw new ArgumentException($"Baseline checkpoint not found: {baselineCheckpointId}");
        }

        // Create current snapshot
        var currentSnapshot = await CreateSnapshotAsync(model, optimizer, cancellationToken);

        // Compute delta
        var delta = ComputeDelta(baselineSnapshot, currentSnapshot);

        // Serialize and compress delta
        var deltaData = SerializeDelta(delta);
        var compressedDelta = await _compressionProvider.CompressAsync(deltaData, cancellationToken);

        // Save delta file
        var deltaPath = await SaveDeltaFileAsync(checkpointId, compressedDelta, cancellationToken);

        // Update snapshot
        _snapshots[checkpointId] = currentSnapshot;

        _logger?.LogInformation(
            "Incremental checkpoint saved: {CheckpointId} (changed tensors: {ChangedCount}, delta size: {DeltaSize} bytes)",
            checkpointId,
            delta.ChangedTensors.Count,
            compressedDelta.Length);

        return checkpointId;
    }

    /// <summary>
    /// Load a checkpoint (full or incremental)
    /// </summary>
    public async Task<CheckpointLoadResult> LoadAsync(
        IStateful model,
        IStateful optimizer,
        string checkpointId,
        LoadOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new LoadOptions();

        _logger?.LogInformation("Loading checkpoint: {CheckpointId}", checkpointId);

        // Check if it's a full checkpoint or incremental
        if (_snapshots.ContainsKey(checkpointId))
        {
            // Full checkpoint - load normally
            return await _checkpoint.LoadAsync(model, optimizer, options, cancellationToken);
        }

        // Try to find and load incremental delta
        var delta = await LoadDeltaAsync(checkpointId, cancellationToken);
        if (delta != null)
        {
            return await LoadIncrementalAsync(model, optimizer, delta, cancellationToken);
        }

        // Not found - try normal load
        return await _checkpoint.LoadAsync(model, optimizer, options, cancellationToken);
    }

    private async Task<IncrementalSnapshot> CreateSnapshotAsync(
        IStateful model,
        IStateful optimizer,
        CancellationToken cancellationToken)
    {
        var modelState = model.GetStateDict();
        var optimizerState = optimizer.GetStateDict();

        var snapshot = new IncrementalSnapshot
        {
            Timestamp = DateTime.UtcNow,
            ModelTensors = new Dictionary<string, TensorSnapshot>()
        };

        foreach (var (name, tensor) in modelState)
        {
            var checksum = await _checksumCalculator.CalculateChecksumAsync(tensor, cancellationToken);
            snapshot.ModelTensors[name] = new TensorSnapshot
            {
                Name = name,
                Shape = tensor.Shape,
                DataType = tensor.DataType,
                Checksum = checksum
            };
        }

        return snapshot;
    }

    private IncrementalDelta ComputeDelta(
        IncrementalSnapshot baseline,
        IncrementalSnapshot current)
    {
        var delta = new IncrementalDelta
        {
            BaselineTimestamp = baseline.Timestamp,
            CurrentTimestamp = current.Timestamp,
            ChangedTensors = new List<TensorDelta>()
        };

        // Find changed tensors
        foreach (var (name, currentTensor) in current.ModelTensors)
        {
            if (!baseline.ModelTensors.TryGetValue(name, out var baselineTensor) ||
                baselineTensor.Checksum != currentTensor.Checksum)
            {
                delta.ChangedTensors.Add(new TensorDelta
                {
                    Name = name,
                    Shape = currentTensor.Shape,
                    DataType = currentTensor.DataType
                });
            }
        }

        return delta;
    }

    private byte[] SerializeDelta(IncrementalDelta delta)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        writer.Write(delta.BaselineTimestamp.Ticks);
        writer.Write(delta.CurrentTimestamp.Ticks);
        writer.Write(delta.ChangedTensors.Count);

        foreach (var tensorDelta in delta.ChangedTensors)
        {
            writer.Write(tensorDelta.Name);
            writer.Write(tensorDelta.Shape.Length);
            foreach (var dim in tensorDelta.Shape)
            {
                writer.Write(dim);
            }
            writer.Write(tensorDelta.DataType);
        }

        return stream.ToArray();
    }

    private IncrementalDelta DeserializeDelta(byte[] data)
    {
        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream);

        var delta = new IncrementalDelta
        {
            BaselineTimestamp = new DateTime(reader.ReadInt64()),
            CurrentTimestamp = new DateTime(reader.ReadInt64())
        };

        var tensorCount = reader.ReadInt32();
        delta.ChangedTensors = new List<TensorDelta>(tensorCount);

        for (int i = 0; i < tensorCount; i++)
        {
            var name = reader.ReadString();
            var shapeLength = reader.ReadInt32();
            var shape = new long[shapeLength];
            for (int j = 0; j < shapeLength; j++)
            {
                shape[j] = reader.ReadInt64();
            }
            var dataType = reader.ReadString();

            delta.ChangedTensors.Add(new TensorDelta
            {
                Name = name,
                Shape = shape,
                DataType = dataType
            });
        }

        return delta;
    }

    private async Task<string> SaveDeltaFileAsync(
        string checkpointId,
        byte[] compressedDelta,
        CancellationToken cancellationToken)
    {
        // Save delta file
        var deltaPath = $"{checkpointId}.delta.zst";
        // Implementation depends on storage backend
        // For now, just return the path
        return deltaPath;
    }

    private async Task<IncrementalDelta?> LoadDeltaAsync(
        string checkpointId,
        CancellationToken cancellationToken)
    {
        // Load delta file
        var deltaPath = $"{checkpointId}.delta.zst";
        // Implementation depends on storage backend
        // For now, return null
        return null;
    }

    private async Task<CheckpointLoadResult> LoadIncrementalAsync(
        IStateful model,
        IStateful optimizer,
        IncrementalDelta delta,
        CancellationToken cancellationToken)
    {
        // Load baseline checkpoint first
        // Then apply delta
        // This is a simplified implementation
        return new CheckpointLoadResult();
    }

    private string GenerateCheckpointId(string type)
    {
        var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
        return $"{type}_{timestamp}";
    }
}
```

### 2. IncrementalSnapshot (Snapshot State)
```csharp
public class IncrementalSnapshot
{
    public DateTime Timestamp { get; set; }
    public Dictionary<string, TensorSnapshot> ModelTensors { get; set; }
    // TODO: Add optimizer state tracking
}
```

### 3. TensorSnapshot (Tensor State)
```csharp
public class TensorSnapshot
{
    public string Name { get; set; }
    public long[] Shape { get; set; }
    public string DataType { get; set; }
    public string Checksum { get; set; }
}
```

### 4. IncrementalDelta (Delta Representation)
```csharp
public class IncrementalDelta
{
    public DateTime BaselineTimestamp { get; set; }
    public DateTime CurrentTimestamp { get; set; }
    public List<TensorDelta> ChangedTensors { get; set; }
}
```

### 5. TensorDelta (Changed Tensor)
```csharp
public class TensorDelta
{
    public string Name { get; set; }
    public long[] Shape { get; set; }
    public string DataType { get; set; }
    // TODO: Add actual tensor data
}
```

### 6. IChecksumCalculator (Interface)
```csharp
public interface IChecksumCalculator
{
    Task<string> CalculateChecksumAsync(Tensor tensor, CancellationToken cancellationToken = default);
}
```

### 7. SHA256ChecksumCalculator (Implementation)
```csharp
public class SHA256ChecksumCalculator : IChecksumCalculator
{
    public async Task<string> CalculateChecksumAsync(
        Tensor tensor,
        CancellationToken cancellationToken = default)
    {
        using var sha256 = SHA256.Create();

        // Get tensor data as bytes
        var data = tensor.GetDataBytes(); // Need to implement this method

        var hashBytes = sha256.ComputeHash(data);
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }
}
```

### 8. ICompressionProvider (Interface)
```csharp
public interface ICompressionProvider
{
    Task<byte[]> CompressAsync(byte[] data, CancellationToken cancellationToken = default);
    Task<byte[]> DecompressAsync(byte[] data, CancellationToken cancellationToken = default);
}
```

### 9. ZstdCompressionProvider (Implementation)
```csharp
public class ZstdCompressionProvider : ICompressionProvider
{
    private readonly int _compressionLevel;

    public ZstdCompressionProvider(int compressionLevel = 3)
    {
        _compressionLevel = compressionLevel;
    }

    public async Task<byte[]> CompressAsync(
        byte[] data,
        CancellationToken cancellationToken = default)
    {
        // TODO: Implement ZSTD compression
        // For now, just return the data
        return await Task.FromResult(data);
    }

    public async Task<byte[]> DecompressAsync(
        byte[] data,
        CancellationToken cancellationToken = default)
    {
        // TODO: Implement ZSTD decompression
        // For now, just return the data
        return await Task.FromResult(data);
    }
}
```

## Usage Examples

### Save Baseline
```csharp
var incrementalManager = new IncrementalCheckpointManager(checkpoint);

// Save baseline checkpoint
await incrementalManager.SaveBaselineAsync(model, optimizer, new SaveOptions
{
    CheckpointPrefix = "baseline_0000"
});
```

### Save Incremental
```csharp
// Save incremental checkpoint (only changed parameters)
await incrementalManager.SaveIncrementalAsync(model, optimizer, "baseline_0000", new SaveOptions
{
    CheckpointPrefix = "incremental_0001"
});
```

### Load Checkpoint
```csharp
// Load checkpoint (works for both full and incremental)
var result = await incrementalManager.LoadAsync(model, optimizer, "incremental_0001");
```

## Benefits
- Reduced storage costs (only save changes)
- Faster checkpoint saves (less I/O)
- Useful for models with sparse updates
- Can reconstruct any checkpoint from baseline + deltas

## Integration Points
- Used by: Training loops for frequent checkpointing
- Depends on: `DistributedCheckpoint`, `IStateful`

## Testing Requirements
- Test baseline snapshot creation
- Test delta computation
- Test delta serialization/deserialization
- Test incremental save/load
- Test checksum calculation
- Test compression/decompression
- Test storage savings

## Success Criteria
- Correctly identifies changed tensors
- Efficiently stores deltas
- Can reconstruct checkpoints from deltas
- Supports multiple incremental checkpoints
- Provides storage savings for sparse updates
