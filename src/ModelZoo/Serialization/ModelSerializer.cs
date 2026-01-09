using System;
using System.IO;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Collections.Generic;
using MLFramework.NN;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;

namespace ModelZoo.Serialization;

/// <summary>
/// Serializes models to the ML Framework native binary format.
/// </summary>
public class ModelSerializer
{
    private readonly SerializerOptions _options;

    /// <summary>
    /// Creates a new ModelSerializer with default options.
    /// </summary>
    public ModelSerializer() : this(new SerializerOptions()) { }

    /// <summary>
    /// Creates a new ModelSerializer with the specified options.
    /// </summary>
    public ModelSerializer(SerializerOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <summary>
    /// Serializes a model to the specified stream.
    /// </summary>
    /// <param name="model">The model to serialize.</param>
    /// <param name="output">The output stream to write to.</param>
    /// <param name="options">Optional serialization options.</param>
    public void Serialize(Module model, Stream output, SerializerOptions options = null)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));
        if (output == null)
            throw new ArgumentNullException(nameof(output));
        if (!output.CanWrite)
            throw new ArgumentException("Stream must be writable.", nameof(output));

        var effectiveOptions = options ?? _options;

        using var writer = new BinaryWriter(output, Encoding.UTF8, leaveOpen: true);

        // Build metadata
        var metadata = new ModelMetadata
        {
            Architecture = model.GetType().Name,
            Layers = new List<LayerInfo>(),
            SerializationTimestamp = DateTime.UtcNow,
            Precision = effectiveOptions.Precision.ToString(),
            Compression = effectiveOptions.Compression.ToString()
        };

        // Collect layer information
        foreach (var (name, param) in model.GetNamedParameters())
        {
            metadata.Layers.Add(new LayerInfo
            {
                Name = name,
                Shape = param.Shape,
                Dtype = param.Dtype.ToString(),
                Size = param.Size
            });
        }

        // Write header
        WriteHeader(writer, effectiveOptions);

        // Write metadata
        WriteMetadata(writer, metadata);

        // Write weights if not metadata only
        if (!effectiveOptions.MetadataOnly)
        {
            WriteWeights(writer, model, effectiveOptions);
        }

        // Write footer
        WriteFooter(writer, output);

        writer.Flush();
    }

    /// <summary>
    /// Serializes a model to the specified file path.
    /// </summary>
    /// <param name="model">The model to serialize.</param>
    /// <param name="path">The file path to write to.</param>
    /// <param name="options">Optional serialization options.</param>
    public void SerializeToFile(Module model, string path, SerializerOptions options = null)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));

        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
        Serialize(model, stream, options);
    }

    /// <summary>
    /// Calculates the size that would be required to serialize the model.
    /// </summary>
    /// <param name="model">The model to calculate size for.</param>
    /// <returns>The size in bytes.</returns>
    public long GetSerializedSize(Module model)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        long size = FileFormatSpec.HeaderSize + FileFormatSpec.FooterSize;

        // Calculate metadata size
        var metadata = BuildMetadata(model);
        var metadataJson = JsonSerializer.Serialize(metadata);
        size += 4 + Encoding.UTF8.GetByteCount(metadataJson); // 4 bytes for length + JSON

        // Calculate weights size
        if (!_options.MetadataOnly)
        {
            size += 4; // NumLayers
            foreach (var (_, param) in model.GetNamedParameters())
            {
                size += 2; // LayerNameLength
                size += Encoding.UTF8.GetByteCount(param.Name); // LayerName
                size += GetTensorDataSize(param, _options.Precision); // TensorData
            }
        }

        return size;
    }

    private void WriteHeader(BinaryWriter writer, SerializerOptions options)
    {
        // Magic bytes
        writer.Write(FileFormatSpec.MagicBytes);

        // Version
        writer.Write(FileFormatSpec.CurrentVersion);

        // Flags
        ushort flags = FileFormatSpec.GetPrecisionFlag(options.Precision) |
                        FileFormatSpec.GetCompressionFlag(options.Compression);
        if (options.IncludeOptimizerState)
            flags |= FileFormatSpec.FlagOptimizerState;
        writer.Write(flags);

        // Header checksum (placeholder for now, could be computed)
        writer.Write((uint)0);

        // Reserved
        writer.Write((uint)0);
    }

    private void WriteMetadata(BinaryWriter writer, ModelMetadata metadata)
    {
        var metadataJson = JsonSerializer.Serialize(metadata);
        var metadataBytes = Encoding.UTF8.GetBytes(metadataJson);

        writer.Write(metadataBytes.Length);
        writer.Write(metadataBytes);
    }

    private void WriteWeights(BinaryWriter writer, Module model, SerializerOptions options)
    {
        var parameters = new List<(string Name, Parameter Parameter)>(model.GetNamedParameters());
        writer.Write(parameters.Count);

        foreach (var (name, param) in parameters)
        {
            // Write layer name
            var nameBytes = Encoding.UTF8.GetBytes(name);
            writer.Write((ushort)nameBytes.Length);
            writer.Write(nameBytes);

            // Write tensor data
            WriteTensorData(writer, param, options.Precision, options.Compression);
        }
    }

    private void WriteTensorData(BinaryWriter writer, Parameter param, SerializationPrecision precision, CompressionType compression)
    {
        var data = ConvertTensorData(param, precision);
        byte[] compressedData = data;

        if (compression != CompressionType.None)
        {
            compressedData = CompressData(data, compression);
        }

        writer.Write(compressedData.Length);
        writer.Write(compressedData);
    }

    private byte[] ConvertTensorData(Parameter param, SerializationPrecision precision)
    {
        if (precision == SerializationPrecision.FP32)
        {
            // Convert float[] to byte[]
            var bytes = new byte[param.Size * sizeof(float)];
            Buffer.BlockCopy(param.Data, 0, bytes, 0, bytes.Length);
            return bytes;
        }
        else if (precision == SerializationPrecision.FP16)
        {
            // Convert to FP16 (simplified - would need proper half-precision conversion)
            // For now, we'll use a simple scaling approach
            var halfData = new byte[param.Size * sizeof(short)];
            for (int i = 0; i < param.Size; i++)
            {
                // Simple truncation for demonstration - real FP16 conversion is more complex
                var value = (float)Math.Clamp(param.Data[i], short.MinValue, short.MaxValue);
                var halfValue = (short)value;
                var halfBytes = BitConverter.GetBytes(halfValue);
                Buffer.BlockCopy(halfBytes, 0, halfData, i * sizeof(short), sizeof(short));
            }
            return halfData;
        }
        else // BF16
        {
            // BF16 conversion (simplified)
            var bfloatData = new byte[param.Size * sizeof(short)];
            for (int i = 0; i < param.Size; i++)
            {
                // Simple truncation for demonstration
                var value = (float)Math.Clamp(param.Data[i], short.MinValue, short.MaxValue);
                var bfloatValue = (short)value;
                var bfloatBytes = BitConverter.GetBytes(bfloatValue);
                Buffer.BlockCopy(bfloatBytes, 0, bfloatData, i * sizeof(short), sizeof(short));
            }
            return bfloatData;
        }
    }

    private byte[] CompressData(byte[] data, CompressionType compression)
    {
        using var output = new MemoryStream();
        if (compression == CompressionType.GZip)
        {
            using (var gzip = new GZipStream(output, CompressionMode.Compress))
            {
                gzip.Write(data, 0, data.Length);
            }
            return output.ToArray();
        }
        else if (compression == CompressionType.Zstd)
        {
            // Zstd would require external library - fallback to GZip for now
            using (var gzip = new GZipStream(output, CompressionMode.Compress))
            {
                gzip.Write(data, 0, data.Length);
            }
            return output.ToArray();
        }

        return data;
    }

    private void WriteFooter(BinaryWriter writer, Stream stream)
    {
        // Compute checksum of the entire file
        var checksum = ComputeChecksum(stream);
        writer.Write(checksum);
    }

    private ulong ComputeChecksum(Stream stream)
    {
        // Simple checksum implementation
        // In production, use CRC64 or SHA256
        using var crc = new System.IO.Hashing.Crc64();
        var position = stream.Position;
        stream.Position = 0;

        var buffer = new byte[8192];
        int bytesRead;
        while ((bytesRead = stream.Read(buffer, 0, buffer.Length)) > 0)
        {
            crc.Append(buffer.AsSpan(0, bytesRead));
        }

        stream.Position = position;
        return crc.GetCurrentHashAsUInt64();
    }

    private ModelMetadata BuildMetadata(Module model)
    {
        var metadata = new ModelMetadata
        {
            Architecture = model.GetType().Name,
            Layers = new List<LayerInfo>(),
            SerializationTimestamp = DateTime.UtcNow,
            Precision = _options.Precision.ToString(),
            Compression = _options.Compression.ToString()
        };

        foreach (var (name, param) in model.GetNamedParameters())
        {
            metadata.Layers.Add(new LayerInfo
            {
                Name = name,
                Shape = param.Shape,
                Dtype = param.Dtype.ToString(),
                Size = param.Size
            });
        }

        return metadata;
    }

    private int GetTensorDataSize(Parameter param, SerializationPrecision precision)
    {
        int bytesPerElement = precision switch
        {
            SerializationPrecision.FP16 => sizeof(short),
            SerializationPrecision.BF16 => sizeof(short),
            _ => sizeof(float)
        };

        return param.Size * bytesPerElement;
    }
}

/// <summary>
/// Model metadata structure for JSON serialization.
/// </summary>
internal class ModelMetadata
{
    public string Architecture { get; set; }
    public List<LayerInfo> Layers { get; set; }
    public DateTime SerializationTimestamp { get; set; }
    public string Precision { get; set; }
    public string Compression { get; set; }
}

/// <summary>
/// Layer information structure for JSON serialization.
/// </summary>
internal class LayerInfo
{
    public string Name { get; set; }
    public int[] Shape { get; set; }
    public string Dtype { get; set; }
    public int Size { get; set; }
}
