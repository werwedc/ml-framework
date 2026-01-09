using System;
using MobileRuntime;

namespace MobileRuntime.Serialization.Models
{
    /// <summary>
    /// Header for the mobile model binary format
    /// </summary>
    public class ModelHeader
    {
        public uint MagicNumber { get; set; }
        public ushort Version { get; set; }
        public uint HeaderChecksum { get; set; }
        public uint Flags { get; set; }
        public uint TotalFileSize { get; set; }
        public uint ModelMetadataOffset { get; set; }
        public uint TensorDataOffset { get; set; }
        public uint OperatorGraphOffset { get; set; }

        public static readonly uint MAGIC_NUMBER = 0x4D4F4249; // "MOBI"
        public static readonly ushort CURRENT_VERSION = 1;
    }

    /// <summary>
    /// Flags used in the model header
    /// </summary>
    [Flags]
    public enum ModelFlags : uint
    {
        None = 0,
        IsQuantized = 1 << 0,
        UsesFP16Weights = 1 << 1,
        GraphOptimized = 1 << 2,
        // Bits 3-31 reserved for future use
    }
}
