# Spec: Multi-Shard Checkpoint Format

## Overview
Implement a multi-shard checkpoint format where each rank writes its own shard file independently. Suitable for large models and parallel I/O optimization.

## Scope
- 30-45 minutes coding time
- Focus on parallel file handling
- Target: `src/MLFramework/Checkpointing/Formats/`

## Classes

### 1. MultiShardCheckpointFormat (Implementation)
```csharp
public class MultiShardCheckpointFormat : ICheckpointFormat
{
    public string Extension => ".shard";

    public Task<byte[]> SerializeAsync(
        List<ShardData> shards,
        CheckpointMetadata metadata,
        CancellationToken cancellationToken)
    {
        // Not used for multi-shard - each shard is written separately
        throw new NotSupportedException("Multi-shard format writes shards individually");
    }

    public Task<(List<ShardData>, CheckpointMetadata)> DeserializeAsync(
        byte[] data,
        CancellationToken cancellationToken)
    {
        // Not used for multi-shard - each shard is read separately
        throw new NotSupportedException("Multi-shard format reads shards individually");
    }

    /// <summary>
    /// Serialize a single shard
    /// </summary>
    public async Task<byte[]> SerializeShardAsync(
        ShardData shard,
        ShardMetadata shardMeta,
        CancellationToken cancellationToken)
    {
        using var memoryStream = new MemoryStream();
        using var writer = new BinaryWriter(memoryStream);

        // Write shard header
        WriteShardHeader(writer, shardMeta);

        // Write tensor data
        WriteTensorData(writer, shard);

        return memoryStream.ToArray();
    }

    /// <summary>
    /// Deserialize a single shard
    /// </summary>
    public async Task<ShardData> DeserializeShardAsync(
        byte[] data,
        ShardMetadata shardMeta,
        CancellationToken cancellationToken)
    {
        using var memoryStream = new MemoryStream(data);
        using var reader = new BinaryReader(memoryStream);

        // Verify shard header
        VerifyShardHeader(reader, shardMeta);

        // Read tensor data
        return ReadTensorData(reader, shardMeta);
    }

    private void WriteShardHeader(BinaryWriter writer, ShardMetadata shardMeta)
    {
        // Write magic number
        writer.Write(0x53484152); // "SHAR" in hex

        // Write rank
        writer.Write(shardMeta.Rank);

        // Write checksum
        var checksumBytes = Encoding.UTF8.GetBytes(shardMeta.Checksum ?? string.Empty);
        writer.Write(checksumBytes.Length);
        writer.Write(checksumBytes);

        // Write tensor count
        writer.Write(shardMeta.Tensors.Count);
    }

    private void VerifyShardHeader(BinaryReader reader, ShardMetadata shardMeta)
    {
        // Verify magic number
        var magic = reader.ReadInt32();
        if (magic != 0x53484152)
        {
            throw new InvalidDataException($"Invalid shard file: magic number mismatch for rank {shardMeta.Rank}");
        }

        // Verify rank
        var rank = reader.ReadInt32();
        if (rank != shardMeta.Rank)
        {
            throw new InvalidDataException($"Shard rank mismatch: expected {shardMeta.Rank}, found {rank}");
        }

        // Read and verify checksum
        var checksumLength = reader.ReadInt32();
        var checksumBytes = reader.ReadBytes(checksumLength);
        var checksum = Encoding.UTF8.GetString(checksumBytes);

        if (checksum != shardMeta.Checksum)
        {
            throw new InvalidDataException($"Shard checksum mismatch for rank {shardMeta.Rank}");
        }

        // Read tensor count
        var tensorCount = reader.ReadInt32();
        if (tensorCount != shardMeta.Tensors.Count)
        {
            throw new InvalidDataException($"Tensor count mismatch for shard {shardMeta.Rank}");
        }
    }

    private void WriteTensorData(BinaryWriter writer, ShardData shard)
    {
        // Note: Actual tensor data comes from the shard.Data byte array
        // This is a simplified version - in practice, you'd iterate through
        // the tensors in the shard and write them with their metadata

        // For now, just write the raw data
        writer.Write(shard.Data.Length);
        writer.Write(shard.Data);
    }

    private ShardData ReadTensorData(BinaryReader reader, ShardMetadata shardMeta)
    {
        var shard = new ShardData();

        // Read raw data
        var dataSize = reader.ReadInt32();
        shard.Data = reader.ReadBytes(dataSize);

        // Copy tensor metadata from metadata
        shard.TensorInfo = new List<TensorMetadata>(shardMeta.Tensors);

        return shard;
    }
}
```

### 2. MultiShardSaver (Save Operation)
```csharp
public class MultiShardSaver
{
    private readonly MultiShardCheckpointFormat _format;
    private readonly ICheckpointStorage _storage;

    public MultiShardSaver(
        MultiShardCheckpointFormat format,
        ICheckpointStorage storage)
    {
        _format = format;
        _storage = storage;
    }

    /// <summary>
    /// Save all shards (called by each rank for its local shard)
    /// </summary>
    public async Task<string> SaveShardAsync(
        string checkpointPrefix,
        int rank,
        ShardData shard,
        ShardMetadata shardMeta,
        CancellationToken cancellationToken = default)
    {
        // Serialize shard
        var data = await _format.SerializeShardAsync(shard, shardMeta, cancellationToken);

        // Write to storage
        var shardPath = $"{checkpointPrefix}_shard_{rank}.shard";
        await _storage.WriteAsync(shardPath, data, cancellationToken);

        return shardPath;
    }

    /// <summary>
    /// Save metadata file (called by rank 0 only)
    /// </summary>
    public async Task<string> SaveMetadataAsync(
        string checkpointPrefix,
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default)
    {
        var metadataJson = MetadataSerializer.Serialize(metadata);
        var metadataBytes = Encoding.UTF8.GetBytes(metadataJson);

        var metadataPath = $"{checkpointPrefix}.metadata.json";
        await _storage.WriteAsync(metadataPath, metadataBytes, cancellationToken);

        return metadataPath;
    }
}
```

### 3. MultiShardLoader (Load Operation)
```csharp
public class MultiShardLoader
{
    private readonly MultiShardCheckpointFormat _format;
    private readonly ICheckpointStorage _storage;

    public MultiShardLoader(
        MultiShardCheckpointFormat format,
        ICheckpointStorage storage)
    {
        _format = format;
        _storage = storage;
    }

    /// <summary>
    /// Load a specific shard
    /// </summary>
    public async Task<ShardData> LoadShardAsync(
        string checkpointPrefix,
        int rank,
        ShardMetadata shardMeta,
        CancellationToken cancellationToken = default)
    {
        // Read from storage
        var shardPath = $"{checkpointPrefix}_shard_{rank}.shard";
        var data = await _storage.ReadAsync(shardPath, cancellationToken);

        // Deserialize
        return await _format.DeserializeShardAsync(data, shardMeta, cancellationToken);
    }

    /// <summary>
    /// Load metadata file
    /// </summary>
    public async Task<CheckpointMetadata> LoadMetadataAsync(
        string checkpointPrefix,
        CancellationToken cancellationToken = default)
    {
        // Read from storage
        var metadataPath = $"{checkpointPrefix}.metadata.json";
        var data = await _storage.ReadAsync(metadataPath, cancellationToken);

        // Deserialize
        var metadataJson = Encoding.UTF8.GetString(data);
        return MetadataSerializer.Deserialize(metadataJson);
    }

    /// <summary>
    /// Load all shards (for consolidation or validation)
    /// </summary>
    public async Task<List<ShardData>> LoadAllShardsAsync(
        string checkpointPrefix,
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default)
    {
        var shards = new List<ShardData>();

        foreach (var shardMeta in metadata.Shards)
        {
            var shard = await LoadShardAsync(
                checkpointPrefix,
                shardMeta.Rank,
                shardMeta,
                cancellationToken);
            shards.Add(shard);
        }

        return shards;
    }
}
```

### 4. ShardFileVerifier (Integrity Verification)
```csharp
public class ShardFileVerifier
{
    private readonly ICheckpointStorage _storage;

    public ShardFileVerifier(ICheckpointStorage storage)
    {
        _storage = storage;
    }

    /// <summary>
    /// Verify all shard files exist and match metadata
    /// </summary>
    public async Task<VerificationResult> VerifyCheckpointAsync(
        string checkpointPrefix,
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default)
    {
        var result = new VerificationResult();

        foreach (var shardMeta in metadata.Shards)
        {
            var shardPath = $"{checkpointPrefix}_shard_{shardMeta.Rank}.shard";

            // Check existence
            var exists = await _storage.ExistsAsync(shardPath, cancellationToken);
            if (!exists)
            {
                result.AddError($"Shard file not found: {shardPath}");
                continue;
            }

            // Check file size
            var fileMetadata = await _storage.GetMetadataAsync(shardPath, cancellationToken);
            if (fileMetadata.Size != shardMeta.FileSize)
            {
                result.AddError($"Shard file size mismatch: {shardPath} (expected {shardMeta.FileSize}, found {fileMetadata.Size})");
            }

            // Verify checksum (optional - can be deferred)
            if (!string.IsNullOrEmpty(shardMeta.Checksum))
            {
                var data = await _storage.ReadAsync(shardPath, cancellationToken);
                var computedChecksum = ComputeChecksum(data);
                if (computedChecksum != shardMeta.Checksum)
                {
                    result.AddError($"Shard checksum mismatch: {shardPath}");
                }
            }
        }

        // Verify metadata file
        var metadataPath = $"{checkpointPrefix}.metadata.json";
        var metadataExists = await _storage.ExistsAsync(metadataPath, cancellationToken);
        if (!metadataExists)
        {
            result.AddError($"Metadata file not found: {metadataPath}");
        }

        return result;
    }

    private string ComputeChecksum(byte[] data)
    {
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(data);
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }
}
```

### 5. VerificationResult (Verification Output)
```csharp
public class VerificationResult
{
    public List<string> Errors { get; } = new();
    public List<string> Warnings { get; } = new();

    public bool IsValid => Errors.Count == 0;

    public void AddError(string error) => Errors.Add(error);
    public void AddWarning(string warning) => Warnings.Add(warning);
}
```

## File Naming Convention
- Shard N: `{checkpoint_prefix}_shard_{N}.shard`
- Metadata: `{checkpoint_prefix}.metadata.json`

## File Structure

### Shard File
```
[Header: 4 bytes]    - Magic number: 0x53484152 ("SHAR")
[Rank: 4 bytes]       - Rank number
[ChecksumLen: 4 bytes] - Checksum length
[Checksum: variable]   - SHA-256 checksum
[TensorCount: 4 bytes] - Number of tensors
[DataLen: 8 bytes]     - Data length
[Data: variable]       - Tensor data
```

## Integration Points
- Used by: `DistributedCheckpointCoordinator`, `CheckpointLoader`
- Depends on: `ICheckpointStorage`, `CheckpointMetadata`, `ShardData`

## Testing Requirements
- Test shard serialization/deserialization
- Test parallel shard writes
- Test metadata file handling
- Test shard file verification
- Test checksum validation
- Test with multiple shards

## Success Criteria
- Each rank writes its shard independently
- Metadata tracks all shard locations
- Checksum validation catches corruption
- Parallel I/O improves performance
- Shard files can be loaded individually or together
