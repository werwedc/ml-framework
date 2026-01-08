namespace MachineLearning.Checkpointing;

using System.Text;
using System.Text.Json;

/// <summary>
/// Single-file checkpoint format implementation
/// All shards are consolidated into a single contiguous file
/// </summary>
public class SingleFileCheckpointFormat : ICheckpointFormat
{
    /// <summary>
    /// File extension for single-file checkpoints
    /// </summary>
    public string Extension => ".checkpoint";

    /// <summary>
    /// Magic number for checkpoint file format validation
    /// "MLCP" in hex (0x4D4C4350)
    /// </summary>
    private const int MagicNumber = 0x4D4C4350;

    /// <summary>
    /// Serialize shards and metadata into a single byte array
    /// </summary>
    public async Task<byte[]> SerializeAsync(
        List<ShardData> shards,
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default)
    {
        if (shards == null)
            throw new ArgumentNullException(nameof(shards));

        if (metadata == null)
            throw new ArgumentNullException(nameof(metadata));

        cancellationToken.ThrowIfCancellationRequested();

        using var memoryStream = new MemoryStream();
        using var writer = new BinaryWriter(memoryStream, Encoding.UTF8, leaveOpen: true);

        // Write header
        WriteHeader(writer, metadata);

        // Write tensor data for each shard
        foreach (var shard in shards)
        {
            WriteShard(writer, shard);
        }

        writer.Flush();
        return await Task.FromResult(memoryStream.ToArray());
    }

    /// <summary>
    /// Deserialize byte array into shards and metadata
    /// </summary>
    public async Task<(List<ShardData>, CheckpointMetadata)> DeserializeAsync(
        byte[] data,
        CancellationToken cancellationToken = default)
    {
        if (data == null || data.Length == 0)
            throw new ArgumentException("Data cannot be empty", nameof(data));

        cancellationToken.ThrowIfCancellationRequested();

        using var memoryStream = new MemoryStream(data);
        using var reader = new BinaryReader(memoryStream, Encoding.UTF8, leaveOpen: true);

        // Read header and metadata
        var metadata = ReadHeader(reader);

        // Read tensor data for each shard
        var shards = new List<ShardData>();
        if (metadata.Shards != null)
        {
            foreach (var shardMeta in metadata.Shards)
            {
                var shard = ReadShard(reader, shardMeta);
                shards.Add(shard);
            }
        }

        return await Task.FromResult((shards, metadata));
    }

    /// <summary>
    /// Write the checkpoint file header
    /// </summary>
    private void WriteHeader(BinaryWriter writer, CheckpointMetadata metadata)
    {
        // Write magic number
        writer.Write(MagicNumber);

        // Write version
        writer.Write(metadata.Version ?? "1.0.0");

        // Write metadata as JSON
        var metadataJson = MetadataSerializer.Serialize(metadata);
        var metadataBytes = Encoding.UTF8.GetBytes(metadataJson);
        writer.Write(metadataBytes.Length);
        writer.Write(metadataBytes);
    }

    /// <summary>
    /// Read and validate the checkpoint file header
    /// </summary>
    private CheckpointMetadata ReadHeader(BinaryReader reader)
    {
        // Verify magic number
        var magic = reader.ReadInt32();
        if (magic != MagicNumber)
        {
            throw new InvalidDataException($"Invalid checkpoint file: magic number mismatch (expected 0x{MagicNumber:X8}, found 0x{magic:X8})");
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

    /// <summary>
    /// Write a shard's tensor data to the file
    /// </summary>
    private void WriteShard(BinaryWriter writer, ShardData shard)
    {
        if (shard == null)
            throw new ArgumentNullException(nameof(shard));

        // Write number of tensors in this shard
        writer.Write(shard.TensorInfo.Count);

        // Write each tensor
        foreach (var tensorInfo in shard.TensorInfo)
        {
            WriteTensor(writer, tensorInfo, shard);
        }
    }

    /// <summary>
    /// Read a shard's tensor data from the file
    /// </summary>
    private ShardData ReadShard(BinaryReader reader, ShardMetadata shardMeta)
    {
        if (shardMeta == null)
            throw new ArgumentNullException(nameof(shardMeta));

        var shard = new ShardData
        {
            Rank = shardMeta.Rank,
            Data = Array.Empty<byte>(),
            TensorInfo = new List<TensorMetadata>()
        };

        // Read number of tensors in this shard
        var tensorCount = reader.ReadInt32();

        // Read each tensor
        for (int i = 0; i < tensorCount; i++)
        {
            var tensor = ReadTensor(reader, shardMeta);
            shard.TensorInfo.Add(tensor);
        }

        return shard;
    }

    /// <summary>
    /// Write a tensor's metadata and data to the file
    /// </summary>
    private void WriteTensor(BinaryWriter writer, TensorMetadata tensorInfo, ShardData shard)
    {
        if (tensorInfo == null)
            throw new ArgumentNullException(nameof(tensorInfo));

        // Write tensor name length and name
        var nameBytes = Encoding.UTF8.GetBytes(tensorInfo.Name);
        writer.Write(nameBytes.Length);
        writer.Write(nameBytes);

        // Write tensor metadata
        writer.Write(tensorInfo.DataType.ToString());
        writer.Write(tensorInfo.Shape.Length);
        foreach (var dim in tensorInfo.Shape)
        {
            writer.Write(dim);
        }

        // Write tensor data size
        writer.Write(tensorInfo.Size);
    }

    /// <summary>
    /// Read a tensor's metadata from the file
    /// </summary>
    private TensorMetadata ReadTensor(BinaryReader reader, ShardMetadata shardMeta)
    {
        // Read tensor name
        var nameLength = reader.ReadInt32();
        var nameBytes = reader.ReadBytes(nameLength);
        var name = Encoding.UTF8.GetString(nameBytes);

        // Read tensor data type
        var dataTypeString = reader.ReadString();
        if (!Enum.TryParse<TensorDataType>(dataTypeString, out var dataType))
        {
            throw new InvalidDataException($"Invalid tensor data type: {dataTypeString}");
        }

        // Read tensor shape
        var shapeLength = reader.ReadInt32();
        var shape = new long[shapeLength];
        for (int i = 0; i < shapeLength; i++)
        {
            shape[i] = reader.ReadInt64();
        }

        // Read tensor size
        var size = reader.ReadInt64();

        return new TensorMetadata
        {
            Name = name,
            DataType = dataType,
            Shape = shape,
            Size = size,
            Offset = 0 // Will be set by the loader if needed
        };
    }
}
