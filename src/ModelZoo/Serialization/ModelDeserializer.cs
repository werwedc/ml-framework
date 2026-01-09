using System;
using System.IO;
using System.IO.Compression;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using MLFramework.NN;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;

namespace ModelZoo.Serialization;

/// <summary>
/// Deserializes models from the ML Framework native binary format.
/// </summary>
public class ModelDeserializer
{
    /// <summary>
    /// Deserializes a model from the specified stream.
    /// </summary>
    /// <param name="input">The input stream to read from.</param>
    /// <param name="device">Optional device to load tensors on (for future implementation).</param>
    /// <returns>A dictionary mapping parameter names to their tensors.</returns>
    public Dictionary<string, Parameter> Deserialize(Stream input, object device = null)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (!input.CanRead)
            throw new ArgumentException("Stream must be readable.", nameof(input));

        using var reader = new BinaryReader(input, Encoding.UTF8, leaveOpen: true);

        // Read and validate header
        var (version, flags) = ReadAndValidateHeader(reader);

        // Read metadata
        var metadata = ReadMetadata(reader);

        // Read weights
        var parameters = ReadWeights(reader, metadata, flags);

        // Verify footer
        VerifyFooter(reader, input);

        return parameters;
    }

    /// <summary>
    /// Deserializes a model from the specified file path.
    /// </summary>
    /// <param name="path">The file path to read from.</param>
    /// <param name="device">Optional device to load tensors on.</param>
    /// <returns>A dictionary mapping parameter names to their tensors.</returns>
    public Dictionary<string, Parameter> DeserializeFromFile(string path, object device = null)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));
        if (!File.Exists(path))
            throw new FileNotFoundException($"File not found: {path}", path);

        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
        return Deserialize(stream, device);
    }

    /// <summary>
    /// Verifies the checksum of a model file without loading it.
    /// </summary>
    /// <param name="path">The path to the model file.</param>
    /// <returns>True if the checksum is valid, false otherwise.</returns>
    public bool VerifyChecksum(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));
        if (!File.Exists(path))
            throw new FileNotFoundException($"File not found: {path}", path);

        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        try
        {
            // Skip header
            stream.Seek(FileFormatSpec.HeaderSize, SeekOrigin.Begin);

            // Read metadata length and skip metadata
            int metadataLength = reader.ReadInt32();
            stream.Seek(metadataLength, SeekOrigin.Current);

            // Skip all weights (we don't need to read them)
            int numLayers = reader.ReadInt32();
            for (int i = 0; i < numLayers; i++)
            {
                int nameLength = reader.ReadUInt16();
                stream.Seek(nameLength, SeekOrigin.Current);
                int dataSize = reader.ReadInt32();
                stream.Seek(dataSize, SeekOrigin.Current);
            }

            // Verify footer
            VerifyFooter(reader, stream);

            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Reads only the metadata from a model file without loading weights.
    /// </summary>
    /// <param name="path">The path to the model file.</param>
    /// <returns>The model metadata.</returns>
    public ModelMetadata ReadMetadataOnly(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));
        if (!File.Exists(path))
            throw new FileNotFoundException($"File not found: {path}", path);

        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        // Read and validate header
        ReadAndValidateHeader(reader);

        // Read metadata
        return ReadMetadata(reader);
    }

    private (ushort version, ushort flags) ReadAndValidateHeader(BinaryReader reader)
    {
        var magic = reader.ReadBytes(4);
        if (!FileFormatSpec.ValidateMagicBytes(magic))
            throw new InvalidDataException("Invalid magic bytes. This is not a valid ML Framework model file.");

        ushort version = reader.ReadUInt16();
        if (version > FileFormatSpec.CurrentVersion)
            throw new InvalidDataException($"Unsupported version: {version}. Maximum supported version: {FileFormatSpec.CurrentVersion}");

        ushort flags = reader.ReadUInt16();

        // Skip header checksum and reserved bytes
        reader.ReadUInt32(); // HeaderChecksum
        reader.ReadUInt32(); // Reserved

        return (version, flags);
    }

    private ModelMetadata ReadMetadata(BinaryReader reader)
    {
        int metadataLength = reader.ReadInt32();
        byte[] metadataBytes = reader.ReadBytes(metadataLength);
        string metadataJson = Encoding.UTF8.GetString(metadataBytes);

        return JsonSerializer.Deserialize<ModelMetadata>(metadataJson);
    }

    private Dictionary<string, Parameter> ReadWeights(BinaryReader reader, ModelMetadata metadata, ushort flags)
    {
        var parameters = new Dictionary<string, Parameter>();
        var compression = FileFormatSpec.GetCompressionFromFlags(flags);
        var precision = FileFormatSpec.GetPrecisionFromFlags(flags);

        int numLayers = reader.ReadInt32();

        for (int i = 0; i < numLayers; i++)
        {
            // Read layer name
            int nameLength = reader.ReadUInt16();
            byte[] nameBytes = reader.ReadBytes(nameLength);
            string layerName = Encoding.UTF8.GetString(nameBytes);

            // Read tensor data
            int dataSize = reader.ReadInt32();
            byte[] dataBytes = reader.ReadBytes(dataSize);

            // Decompress if needed
            if (compression != CompressionType.None)
            {
                dataBytes = DecompressData(dataBytes, compression);
            }

            // Convert to tensor
            var layerInfo = metadata.Layers.Find(l => l.Name == layerName);
            if (layerInfo == null)
            {
                throw new InvalidDataException($"Layer '{layerName}' not found in metadata.");
            }

            float[] tensorData = ConvertToFloatArray(dataBytes, layerInfo.Size, precision);
            var tensor = new Tensor(tensorData, layerInfo.Shape, false, DataType.Float32);
            var param = new Parameter(tensor, layerName);

            parameters[layerName] = param;
        }

        return parameters;
    }

    private float[] ConvertToFloatArray(byte[] data, int expectedSize, SerializationPrecision precision)
    {
        float[] floatArray = new float[expectedSize];

        if (precision == SerializationPrecision.FP32)
        {
            Buffer.BlockCopy(data, 0, floatArray, 0, data.Length);
        }
        else if (precision == SerializationPrecision.FP16 || precision == SerializationPrecision.BF16)
        {
            // Convert short[] to float[] (simplified)
            var shortArray = new short[expectedSize];
            Buffer.BlockCopy(data, 0, shortArray, 0, data.Length);

            for (int i = 0; i < expectedSize; i++)
            {
                floatArray[i] = (float)shortArray[i];
            }
        }

        return floatArray;
    }

    private byte[] DecompressData(byte[] compressedData, CompressionType compression)
    {
        if (compression == CompressionType.GZip || compression == CompressionType.Zstd)
        {
            using var input = new MemoryStream(compressedData);
            using var gzip = new GZipStream(input, CompressionMode.Decompress);
            using var output = new MemoryStream();

            gzip.CopyTo(output);
            return output.ToArray();
        }

        return compressedData;
    }

    private void VerifyFooter(BinaryReader reader, Stream stream)
    {
        var position = stream.Position;

        // Read the stored checksum from the end of the file
        stream.Seek(-FileFormatSpec.FooterSize, SeekOrigin.End);
        ulong storedChecksum = reader.ReadUInt64();

        // Calculate current checksum
        var computedChecksum = ComputeChecksum(stream);

        if (storedChecksum != computedChecksum)
            throw new InvalidDataException($"Checksum verification failed. File may be corrupted.");

        // Restore position
        stream.Position = position;
    }

    private ulong ComputeChecksum(Stream stream)
    {
        // Compute checksum of the entire file except the footer
        using var crc = new System.IO.Hashing.Crc64();
        var position = stream.Position;

        // Position at the start of the file (after header if needed)
        stream.Position = 0;

        // Read up to but not including the footer
        long bytesToRead = stream.Length - FileFormatSpec.FooterSize;
        var buffer = new byte[8192];
        int bytesRead;

        while ((bytesRead = stream.Read(buffer, 0, buffer.Length)) > 0)
        {
            if (bytesRead > bytesToRead)
            {
                bytesRead = (int)bytesToRead;
            }
            crc.Append(buffer.AsSpan(0, bytesRead));
            bytesToRead -= bytesRead;

            if (bytesToRead <= 0)
                break;
        }

        stream.Position = position;
        return crc.GetCurrentHashAsUInt64();
    }
}
