# Spec: Single-File Checkpoint Format

## Overview
Implement a single-file checkpoint format where all shards are gathered to rank 0 and written as one contiguous file. Suitable for smaller models or when consolidation speed is not critical.

## Scope
- 30-45 minutes coding time
- Focus on file format and I/O
- Target: `src/MLFramework/Checkpointing/Formats/`

## Classes

### 1. ICheckpointFormat (Interface)
```csharp
public interface ICheckpointFormat
{
    string Extension { get; }

    Task<byte[]> SerializeAsync(
        List<ShardData> shards,
        CheckpointMetadata metadata,
        CancellationToken cancellationToken);

    Task<(List<ShardData>, CheckpointMetadata)> DeserializeAsync(
        byte[] data,
        CancellationToken cancellationToken);
}
```

### 2. SingleFileCheckpointFormat (Implementation)
```csharp
public class SingleFileCheckpointFormat : ICheckpointFormat
{
    public string Extension => ".checkpoint";

    public async Task<byte[]> SerializeAsync(
        List<ShardData> shards,
        CheckpointMetadata metadata,
        CancellationToken cancellationToken)
    {
        using var memoryStream = new MemoryStream();
        using var writer = new BinaryWriter(memoryStream);

        // Write header
        WriteHeader(writer, metadata);

        // Write tensor data
        foreach (var shard in shards)
        {
            WriteShard(writer, shard);
        }

        return memoryStream.ToArray();
    }

    public async Task<(List<ShardData>, CheckpointMetadata)> DeserializeAsync(
        byte[] data,
        CancellationToken cancellationToken)
    {
        using var memoryStream = new MemoryStream(data);
        using var reader = new BinaryReader(memoryStream);

        // Read header
        var metadata = ReadHeader(reader);

        // Read tensor data
        var shards = new List<ShardData>();
        foreach (var shardMeta in metadata.Shards)
        {
            var shard = ReadShard(reader, shardMeta);
            shards.Add(shard);
        }

        return (shards, metadata);
    }

    private void WriteHeader(BinaryWriter writer, CheckpointMetadata metadata)
    {
        // Write magic number
        writer.Write(0x4D4C4350); // "MLCP" in hex

        // Write version
        writer.Write(metadata.Version);

        // Write metadata length and metadata
        var metadataJson = MetadataSerializer.Serialize(metadata);
        var metadataBytes = Encoding.UTF8.GetBytes(metadataJson);
        writer.Write(metadataBytes.Length);
        writer.Write(metadataBytes);
    }

    private CheckpointMetadata ReadHeader(BinaryReader reader)
    {
        // Verify magic number
        var magic = reader.ReadInt32();
        if (magic != 0x4D4C4350)
        {
            throw new InvalidDataException("Invalid checkpoint file: magic number mismatch");
        }

        // Read version
        var version = reader.ReadString();

        // Read metadata
        var metadataLength = reader.ReadInt32();
        var metadataBytes = reader.ReadBytes(metadataLength);
        var metadataJson = Encoding.UTF8.GetString(metadataBytes);
        var metadata = MetadataSerializer.Deserialize(metadataJson);

        return metadata;
    }

    private void WriteShard(BinaryWriter writer, ShardData shard)
    {
        // Write number of tensors
        writer.Write(shard.TensorInfo.Count);

        // Write each tensor
        foreach (var tensorInfo in shard.TensorInfo)
        {
            WriteTensor(writer, tensorInfo);
        }
    }

    private ShardData ReadShard(BinaryReader reader, ShardMetadata shardMeta)
    {
        var shard = new ShardData();

        // Read number of tensors
        var tensorCount = reader.ReadInt32();

        // Read each tensor
        for (int i = 0; i < tensorCount; i++)
        {
            var tensor = ReadTensor(reader);
            shard.TensorInfo.Add(tensor.Metadata);
        }

        return shard;
    }

    private void WriteTensor(BinaryWriter writer, TensorMetadata metadata)
    {
        // Write tensor name length and name
        var nameBytes = Encoding.UTF8.GetBytes(metadata.Name);
        writer.Write(nameBytes.Length);
        writer.Write(nameBytes);

        // Write tensor metadata
        writer.Write(metadata.DataType);
        writer.Write(metadata.Shape.Length);
        foreach (var dim in metadata.Shape)
        {
            writer.Write(dim);
        }

        // Write tensor data (placeholder - actual data comes from elsewhere)
        writer.Write(metadata.Size);
    }

    private Tensor ReadTensor(BinaryReader reader)
    {
        // Read tensor name
        var nameLength = reader.ReadInt32();
        var nameBytes = reader.ReadBytes(nameLength);
        var name = Encoding.UTF8.GetString(nameBytes);

        // Read tensor metadata
        var dataType = reader.ReadString();
        var shapeLength = reader.ReadInt32();
        var shape = new long[shapeLength];
        for (int i = 0; i < shapeLength; i++)
        {
            shape[i] = reader.ReadInt64();
        }

        // Read tensor data size
        var size = reader.ReadInt64();

        return new Tensor
        {
            Metadata = new TensorMetadata
            {
                Name = name,
                DataType = dataType,
                Shape = shape,
                Size = size
            }
        };
    }
}
```

### 3. SingleFileSaver (Save Operation)
```csharp
public class SingleFileSaver
{
    private readonly ICheckpointFormat _format;
    private readonly ICheckpointStorage _storage;

    public SingleFileSaver(
        ICheckpointFormat format,
        ICheckpointStorage storage)
    {
        _format = format;
        _storage = storage;
    }

    public async Task<string> SaveAsync(
        string checkpointPrefix,
        List<ShardData> shards,
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default)
    {
        // Serialize all shards into single file
        var data = await _format.SerializeAsync(shards, metadata, cancellationToken);

        // Write to storage
        var filePath = $"{checkpointPrefix}{_format.Extension}";
        await _storage.WriteAsync(filePath, data, cancellationToken);

        return filePath;
    }
}
```

### 4. SingleFileLoader (Load Operation)
```csharp
public class SingleFileLoader
{
    private readonly ICheckpointFormat _format;
    private readonly ICheckpointStorage _storage;

    public SingleFileLoader(
        ICheckpointFormat format,
        ICheckpointStorage storage)
    {
        _format = format;
        _storage = storage;
    }

    public async Task<(List<ShardData>, CheckpointMetadata)> LoadAsync(
        string checkpointPath,
        CancellationToken cancellationToken = default)
    {
        // Read from storage
        var data = await _storage.ReadAsync(checkpointPath, cancellationToken);

        // Deserialize
        return await _format.DeserializeAsync(data, cancellationToken);
    }
}
```

## File Structure
```
[Header: 4 bytes]    - Magic number: 0x4D4C4350 ("MLCP")
[Version: variable]  - Version string
[MetaLen: 4 bytes]   - Metadata length
[Metadata: variable] - JSON metadata
[Shard1: variable]   - First shard data
[Shard2: variable]   - Second shard data
...
[ShardN: variable]   - Last shard data
```

## Shard Structure
```
[TensorCount: 4 bytes] - Number of tensors
[Tensor1: variable]     - First tensor
[Tensor2: variable]     - Second tensor
...
[TensorN: variable]     - Last tensor
```

## Tensor Structure
```
[NameLen: 4 bytes]  - Name length
[Name: variable]     - Tensor name
[DataType: variable] - Data type string
[ShapeLen: 4 bytes] - Shape array length
[Shape: variable]   - Shape dimensions
[Size: 8 bytes]     - Size in bytes
```

## Integration Points
- Used by: `DistributedCheckpointCoordinator`
- Depends on: `ICheckpointStorage`, `CheckpointMetadata`, `ShardData`

## Testing Requirements
- Test serialize/deserialize roundtrip
- Test with multiple shards
- Test with different tensor shapes and data types
- Test file format validation (magic number)
- Test error handling for corrupted files

## Success Criteria
- Can serialize and deserialize checkpoints correctly
- File format is versioned and extensible
- Handles multiple shards efficiently
- Provides clear error messages for corrupted files
- Suitable for smaller models or consolidated checkpoints
