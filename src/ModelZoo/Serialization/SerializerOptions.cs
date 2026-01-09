namespace ModelZoo.Serialization;

/// <summary>
/// Configuration options for model serialization.
/// </summary>
public class SerializerOptions
{
    /// <summary>
    /// Gets or sets the precision for tensor data.
    /// </summary>
    public SerializationPrecision Precision { get; set; } = SerializationPrecision.FP32;

    /// <summary>
    /// Gets or sets the compression algorithm to use.
    /// </summary>
    public CompressionType Compression { get; set; } = CompressionType.None;

    /// <summary>
    /// Gets or sets whether to include optimizer state in the serialized model.
    /// </summary>
    public bool IncludeOptimizerState { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to serialize only metadata (for inspection).
    /// </summary>
    public bool MetadataOnly { get; set; } = false;

    /// <summary>
    /// Creates a new instance of SerializerOptions with default values.
    /// </summary>
    public SerializerOptions() { }

    /// <summary>
    /// Creates a new instance with the specified precision.
    /// </summary>
    public SerializerOptions(SerializationPrecision precision)
    {
        Precision = precision;
    }

    /// <summary>
    /// Creates a new instance with the specified precision and compression.
    /// </summary>
    public SerializerOptions(SerializationPrecision precision, CompressionType compression)
    {
        Precision = precision;
        Compression = compression;
    }
}

/// <summary>
/// Precision options for serialization.
/// </summary>
public enum SerializationPrecision
{
    /// <summary>
    /// 32-bit floating point (default).
    /// </summary>
    FP32 = 0,

    /// <summary>
    /// 16-bit floating point (IEEE 754 half precision).
    /// </summary>
    FP16 = 1,

    /// <summary>
    /// 16-bit brain floating point (Google's format).
    /// </summary>
    BF16 = 2
}

/// <summary>
/// Compression types for serialization.
/// </summary>
public enum CompressionType
{
    /// <summary>
    /// No compression.
    /// </summary>
    None = 0,

    /// <summary>
    /// GZip compression.
    /// </summary>
    GZip = 1,

    /// <summary>
    /// Zstd compression (if available).
    /// </summary>
    Zstd = 2
}
