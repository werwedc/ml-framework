namespace ModelZoo.Serialization;

/// <summary>
/// File format specification for the ML Framework native serialization format.
/// </summary>
public static class FileFormatSpec
{
    /// <summary>
    /// Magic bytes to identify ML Framework model files.
    /// </summary>
    public static readonly byte[] MagicBytes = new byte[] { 0x4D, 0x4C, 0x46, 0x57 }; // "MLFW"

    /// <summary>
    /// Current version of the file format.
    /// </summary>
    public const ushort CurrentVersion = 1;

    /// <summary>
    /// Size of the header section in bytes.
    /// </summary>
    public const int HeaderSize = 16;

    /// <summary>
    /// Size of the footer section in bytes.
    /// </summary>
    public const int FooterSize = 8;

    /// <summary>
    /// Flag indicating compression is used.
    /// </summary>
    public const ushort FlagCompression = 0x0001;

    /// <summary>
    /// Flag indicating FP16 precision.
    /// </summary>
    public const ushort FlagFP16 = 0x0002;

    /// <summary>
    /// Flag indicating BF16 precision.
    /// </summary>
    public const ushort FlagBF16 = 0x0004;

    /// <summary>
    /// Flag indicating optimizer state is included.
    /// </summary>
    public const ushort FlagOptimizerState = 0x0008;

    /// <summary>
    /// Gets the precision flag based on the specified precision setting.
    /// </summary>
    public static ushort GetPrecisionFlag(SerializationPrecision precision)
    {
        return precision switch
        {
            SerializationPrecision.FP16 => FlagFP16,
            SerializationPrecision.BF16 => FlagBF16,
            _ => 0
        };
    }

    /// <summary>
    /// Gets the compression flag based on the specified compression type.
    /// </summary>
    public static ushort GetCompressionFlag(CompressionType compression)
    {
        return compression switch
        {
            CompressionType.GZip => FlagCompression,
            CompressionType.Zstd => FlagCompression,
            _ => 0
        };
    }

    /// <summary>
    /// Validates that the provided magic bytes match the expected format.
    /// </summary>
    public static bool ValidateMagicBytes(byte[] bytes)
    {
        if (bytes == null || bytes.Length < MagicBytes.Length)
            return false;

        for (int i = 0; i < MagicBytes.Length; i++)
        {
            if (bytes[i] != MagicBytes[i])
                return false;
        }
        return true;
    }

    /// <summary>
    /// Gets the serialization precision from file flags.
    /// </summary>
    public static SerializationPrecision GetPrecisionFromFlags(ushort flags)
    {
        if ((flags & FlagBF16) != 0)
            return SerializationPrecision.BF16;
        if ((flags & FlagFP16) != 0)
            return SerializationPrecision.FP16;
        return SerializationPrecision.FP32;
    }

    /// <summary>
    /// Gets the compression type from file flags.
    /// </summary>
    public static CompressionType GetCompressionFromFlags(ushort flags)
    {
        if ((flags & FlagCompression) == 0)
            return CompressionType.None;
        // Default to GZip for now, could be extended to detect Zstd
        return CompressionType.GZip;
    }

    /// <summary>
    /// Checks if optimizer state is included based on file flags.
    /// </summary>
    public static bool HasOptimizerState(ushort flags)
    {
        return (flags & FlagOptimizerState) != 0;
    }
}
