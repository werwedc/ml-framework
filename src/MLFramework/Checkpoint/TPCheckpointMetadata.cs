using System;
using System.Collections.Generic;
using System.IO;

namespace MLFramework.Checkpoint;

/// <summary>
/// Metadata for tensor-parallel model checkpoints
/// </summary>
public class TPCheckpointMetadata
{
    public string ModelName { get; set; } = "";
    public DateTime SavedAt { get; set; } = DateTime.UtcNow;
    public int TPWorldSize { get; set; }
    public int[] MeshShape { get; set; } = Array.Empty<int>();
    public int Version { get; set; } = 1;
    public Dictionary<string, object> AdditionalInfo { get; set; } = new();

    /// <summary>
    /// Serialize metadata to a binary writer
    /// </summary>
    public void Serialize(BinaryWriter writer)
    {
        writer.Write(ModelName);
        writer.Write(SavedAt.ToBinary());
        writer.Write(TPWorldSize);
        writer.Write(MeshShape.Length);
        foreach (var dim in MeshShape)
        {
            writer.Write(dim);
        }
        writer.Write(Version);
        // Simplified: skip AdditionalInfo for now
    }

    /// <summary>
    /// Deserialize metadata from a binary reader
    /// </summary>
    public static TPCheckpointMetadata Deserialize(BinaryReader reader)
    {
        var metadata = new TPCheckpointMetadata
        {
            ModelName = reader.ReadString(),
            SavedAt = DateTime.FromBinary(reader.ReadInt64()),
            TPWorldSize = reader.ReadInt32(),
            Version = reader.ReadInt32()
        };

        int meshDims = reader.ReadInt32();
        metadata.MeshShape = new int[meshDims];
        for (int i = 0; i < meshDims; i++)
        {
            metadata.MeshShape[i] = reader.ReadInt32();
        }

        return metadata;
    }
}
