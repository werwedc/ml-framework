namespace MachineLearning.Checkpointing;

using System.Text;

/// <summary>
/// Multi-shard checkpoint format implementation
/// Each shard writes its own file independently for parallel I/O optimization
/// </summary>
public class MultiShardCheckpointFormat : ICheckpointFormat
{
    /// <summary>
    /// File extension for multi-shard checkpoints
    /// </summary>
    public string Extension => ".shard";

    /// <summary>
    /// Magic number for shard file format validation
    /// "SHAR" in hex (0x53484152)
    /// </summary>
    private const int MagicNumber = 0x53484152;

    /// <summary>
    /// Serialize shards and metadata into a byte array
    /// Not used for multi-shard - each shard is written separately
    /// </summary>
    public Task<byte[]> SerializeAsync(
        List<ShardData> shards,
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default)
    {
        throw new NotSupportedException("Multi-shard format writes shards individually. Use SerializeShardAsync instead.");
    }

    /// <summary>
    /// Deserialize byte array into shards and metadata
    /// Not used for multi-shard - each shard is read separately
    /// </summary>
    public Task<(List<ShardData>, CheckpointMetadata)> DeserializeAsync(
        byte[] data,
        CancellationToken cancellationToken = default)
    {
        throw new NotSupportedException("Multi-shard format reads shards individually. Use DeserializeShardAsync instead.");
    }

    /// <summary>
    /// Serialize a single shard to a byte array
    /// </summary>
    /// <param name="shard">Shard data to serialize</param>
    /// <param name="shardMeta">Shard metadata</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Serialized byte array</returns>
    public async Task<byte[]> SerializeShardAsync(
        ShardData shard,
        ShardMetadata shardMeta,
        CancellationToken cancellationToken = default)
    {
        if (shard == null)
            throw new ArgumentNullException(nameof(shard));

        if (shardMeta == null)
            throw new ArgumentNullException(nameof(shardMeta));

        cancellationToken.ThrowIfCancellationRequested();

        using var memoryStream = new MemoryStream();
        using var writer = new BinaryWriter(memoryStream, Encoding.UTF8, leaveOpen: true);

        // Write shard header
        WriteShardHeader(writer, shardMeta);

        // Write tensor data
        WriteTensorData(writer, shard);

        writer.Flush();
        return await Task.FromResult(memoryStream.ToArray());
    }

    /// <summary>
    /// Deserialize a single shard from a byte array
    /// </summary>
    /// <param name="data">Byte array to deserialize</param>
    /// <param name="shardMeta">Shard metadata</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Deserialized shard data</returns>
    public async Task<ShardData> DeserializeShardAsync(
        byte[] data,
        ShardMetadata shardMeta,
        CancellationToken cancellationToken = default)
    {
        if (data == null || data.Length == 0)
            throw new ArgumentException("Data cannot be empty", nameof(data));

        if (shardMeta == null)
            throw new ArgumentNullException(nameof(shardMeta));

        cancellationToken.ThrowIfCancellationRequested();

        using var memoryStream = new MemoryStream(data);
        using var reader = new BinaryReader(memoryStream, Encoding.UTF8, leaveOpen: true);

        // Verify shard header
        VerifyShardHeader(reader, shardMeta);

        // Read tensor data
        return await Task.FromResult(ReadTensorData(reader, shardMeta));
    }

    /// <summary>
    /// Write shard header to binary stream
    /// </summary>
    private void WriteShardHeader(BinaryWriter writer, ShardMetadata shardMeta)
    {
        // Write magic number
        writer.Write(MagicNumber);

        // Write rank
        writer.Write(shardMeta.Rank);

        // Write checksum
        var checksumBytes = Encoding.UTF8.GetBytes(shardMeta.Checksum ?? string.Empty);
        writer.Write(checksumBytes.Length);
        writer.Write(checksumBytes);

        // Write tensor count
        var tensorCount = shardMeta.Tensors?.Count ?? 0;
        writer.Write(tensorCount);
    }

    /// <summary>
    /// Verify shard header from binary stream
    /// </summary>
    private void VerifyShardHeader(BinaryReader reader, ShardMetadata shardMeta)
    {
        // Verify magic number
        var magic = reader.ReadInt32();
        if (magic != MagicNumber)
        {
            throw new InvalidDataException(
                $"Invalid shard file: magic number mismatch for rank {shardMeta.Rank} " +
                $"(expected 0x{MagicNumber:X8}, found 0x{magic:X8})");
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

        if (!string.IsNullOrEmpty(shardMeta.Checksum) && checksum != shardMeta.Checksum)
        {
            throw new InvalidDataException($"Shard checksum mismatch for rank {shardMeta.Rank}");
        }

        // Read tensor count
        var tensorCount = reader.ReadInt32();
        var expectedTensorCount = shardMeta.Tensors?.Count ?? 0;
        if (tensorCount != expectedTensorCount)
        {
            throw new InvalidDataException(
                $"Tensor count mismatch for shard {shardMeta.Rank}: " +
                $"expected {expectedTensorCount}, found {tensorCount}");
        }
    }

    /// <summary>
    /// Write tensor data to binary stream
    /// </summary>
    private void WriteTensorData(BinaryWriter writer, ShardData shard)
    {
        // Note: Actual tensor data comes from the shard.Data byte array
        // This is a simplified version - in practice, you'd iterate through
        // the tensors in the shard and write them with their metadata

        // For now, just write the raw data
        writer.Write(shard.Data.Length);
        writer.Write(shard.Data);
    }

    /// <summary>
    /// Read tensor data from binary stream
    /// </summary>
    private ShardData ReadTensorData(BinaryReader reader, ShardMetadata shardMeta)
    {
        var shard = new ShardData
        {
            Rank = shardMeta.Rank
        };

        // Read raw data
        var dataSize = reader.ReadInt32();
        shard.Data = reader.ReadBytes(dataSize);

        // Copy tensor metadata from metadata
        if (shardMeta.Tensors != null)
        {
            shard.TensorInfo = new List<TensorMetadata>(shardMeta.Tensors);
        }

        return shard;
    }
}
