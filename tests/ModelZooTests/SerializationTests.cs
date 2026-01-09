using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using MLFramework.NN;
using ModelZoo.Serialization;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;

namespace ModelZooTests;

/// <summary>
/// Tests for model serialization and deserialization.
/// </summary>
public class SerializationTests
{
    [Fact]
    public void SerializeAndDeserialize_SimpleModel_WorksCorrectly()
    {
        // Arrange
        var model = CreateSimpleModel();
        var serializer = new ModelSerializer();
        var deserializer = new ModelDeserializer();

        using var memoryStream = new MemoryStream();

        // Act
        serializer.Serialize(model, memoryStream);
        memoryStream.Position = 0;

        var deserializedParams = deserializer.Deserialize(memoryStream);

        // Assert
        Assert.Equal(2, deserializedParams.Count);
        Assert.True(CompareParameters(model.GetNamedParameters().ToList(), deserializedParams.ToList()));
    }

    [Fact]
    public void SerializeToFileAndDeserializeFromFile_WorksCorrectly()
    {
        // Arrange
        var model = CreateSimpleModel();
        var serializer = new ModelSerializer();
        var deserializer = new ModelDeserializer();
        var testFilePath = Path.Combine(Path.GetTempPath(), "test_model.mlfw");

        try
        {
            // Act
            serializer.SerializeToFile(model, testFilePath);
            var deserializedParams = deserializer.DeserializeFromFile(testFilePath);

            // Assert
            Assert.Equal(2, deserializedParams.Count);
            Assert.True(CompareParameters(model.GetNamedParameters().ToList(), deserializedParams.ToList()));
        }
        finally
        {
            if (File.Exists(testFilePath))
                File.Delete(testFilePath);
        }
    }

    [Fact]
    public void SerializeWithFP16Precision_WorksCorrectly()
    {
        // Arrange
        var model = CreateSimpleModel();
        var options = new SerializerOptions(SerializationPrecision.FP16);
        var serializer = new ModelSerializer(options);
        var deserializer = new ModelDeserializer();

        using var memoryStream = new MemoryStream();

        // Act
        serializer.Serialize(model, memoryStream, options);
        memoryStream.Position = 0;

        var deserializedParams = deserializer.Deserialize(memoryStream);

        // Assert
        Assert.Equal(2, deserializedParams.Count);
        // Note: FP16 will have some precision loss, so we check approximate equality
        foreach (var (originalName, originalParam) in model.GetNamedParameters())
        {
            Assert.True(deserializedParams.ContainsKey(originalName));
            var deserializedParam = deserializedParams[originalName];
            Assert.Equal(originalParam.Shape, deserializedParam.Shape);

            for (int i = 0; i < originalParam.Size; i++)
            {
                Assert.True(Math.Abs(originalParam.Data[i] - deserializedParam.Data[i]) < 0.1,
                    $"Value mismatch at index {i}: {originalParam.Data[i]} vs {deserializedParam.Data[i]}");
            }
        }
    }

    [Fact]
    public void SerializeWithBF16Precision_WorksCorrectly()
    {
        // Arrange
        var model = CreateSimpleModel();
        var options = new SerializerOptions(SerializationPrecision.BF16);
        var serializer = new ModelSerializer(options);
        var deserializer = new ModelDeserializer();

        using var memoryStream = new MemoryStream();

        // Act
        serializer.Serialize(model, memoryStream, options);
        memoryStream.Position = 0;

        var deserializedParams = deserializer.Deserialize(memoryStream);

        // Assert
        Assert.Equal(2, deserializedParams.Count);
        // Note: BF16 will have some precision loss, so we check approximate equality
        foreach (var (originalName, originalParam) in model.GetNamedParameters())
        {
            Assert.True(deserializedParams.ContainsKey(originalName));
            var deserializedParam = deserializedParams[originalName];
            Assert.Equal(originalParam.Shape, deserializedParam.Shape);

            for (int i = 0; i < originalParam.Size; i++)
            {
                Assert.True(Math.Abs(originalParam.Data[i] - deserializedParam.Data[i]) < 0.1,
                    $"Value mismatch at index {i}: {originalParam.Data[i]} vs {deserializedParam.Data[i]}");
            }
        }
    }

    [Fact]
    public void SerializeWithGZipCompression_WorksCorrectly()
    {
        // Arrange
        var model = CreateSimpleModel();
        var options = new SerializerOptions(SerializationPrecision.FP32, CompressionType.GZip);
        var serializer = new ModelSerializer(options);
        var deserializer = new ModelDeserializer();

        using var memoryStream = new MemoryStream();

        // Act
        serializer.Serialize(model, memoryStream, options);
        memoryStream.Position = 0;

        var deserializedParams = deserializer.Deserialize(memoryStream);

        // Assert
        Assert.Equal(2, deserializedParams.Count);
        Assert.True(CompareParameters(model.GetNamedParameters().ToList(), deserializedParams.ToList()));
    }

    [Fact]
    public void SerializeMetadataOnly_WorksCorrectly()
    {
        // Arrange
        var model = CreateSimpleModel();
        var options = new SerializerOptions { MetadataOnly = true };
        var serializer = new ModelSerializer(options);
        var deserializer = new ModelDeserializer();

        using var memoryStream = new MemoryStream();

        // Act
        serializer.Serialize(model, memoryStream, options);
        memoryStream.Position = 0;

        var metadata = deserializer.ReadMetadataOnly(memoryStream);

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal("SimpleModel", metadata.Architecture);
        Assert.Equal(2, metadata.Layers.Count);
        Assert.Equal(SerializationPrecision.FP32.ToString(), metadata.Precision);
    }

    [Fact]
    public void VerifyChecksum_ValidFile_ReturnsTrue()
    {
        // Arrange
        var model = CreateSimpleModel();
        var serializer = new ModelSerializer();
        var deserializer = new ModelDeserializer();
        var testFilePath = Path.Combine(Path.GetTempPath(), "test_model_checksum.mlfw");

        try
        {
            // Act
            serializer.SerializeToFile(model, testFilePath);
            var isValid = deserializer.VerifyChecksum(testFilePath);

            // Assert
            Assert.True(isValid);
        }
        finally
        {
            if (File.Exists(testFilePath))
                File.Delete(testFilePath);
        }
    }

    [Fact]
    public void VerifyChecksum_CorruptedFile_ReturnsFalse()
    {
        // Arrange
        var model = CreateSimpleModel();
        var serializer = new ModelSerializer();
        var deserializer = new ModelDeserializer();
        var testFilePath = Path.Combine(Path.GetTempPath(), "test_model_corrupt.mlfw");

        try
        {
            serializer.SerializeToFile(model, testFilePath);

            // Corrupt the file by modifying some bytes
            using var fileStream = new FileStream(testFilePath, FileMode.Open, FileAccess.ReadWrite);
            fileStream.Seek(100, SeekOrigin.Begin);
            fileStream.WriteByte(0xFF);
            fileStream.WriteByte(0xFF);

            // Act
            var isValid = deserializer.VerifyChecksum(testFilePath);

            // Assert
            Assert.False(isValid);
        }
        finally
        {
            if (File.Exists(testFilePath))
                File.Delete(testFilePath);
        }
    }

    [Fact]
    public void Deserialize_InvalidMagicBytes_ThrowsInvalidDataException()
    {
        // Arrange
        var deserializer = new ModelDeserializer();
        using var memoryStream = new MemoryStream();

        // Write invalid magic bytes
        var writer = new BinaryWriter(memoryStream);
        writer.Write(new byte[] { 0x00, 0x00, 0x00, 0x00 }); // Invalid magic
        writer.Write((ushort)1); // Version
        writer.Write((ushort)0); // Flags
        writer.Write((uint)0); // HeaderChecksum
        writer.Write((uint)0); // Reserved
        writer.Write(0); // MetadataLength
        writer.Write((ulong)0); // FooterChecksum

        memoryStream.Position = 0;

        // Act & Assert
        Assert.Throws<InvalidDataException>(() => deserializer.Deserialize(memoryStream));
    }

    [Fact]
    public void GetSerializedSize_ReturnsCorrectSize()
    {
        // Arrange
        var model = CreateSimpleModel();
        var serializer = new ModelSerializer();

        // Act
        long size = serializer.GetSerializedSize(model);

        // Assert
        Assert.True(size > 0);
        // Size should be at least header + footer size
        Assert.True(size >= FileFormatSpec.HeaderSize + FileFormatSpec.FooterSize);
    }

    [Fact]
    public void SerializeAndDeserialize_LargerModel_WorksCorrectly()
    {
        // Arrange
        var model = CreateLargerModel();
        var serializer = new ModelSerializer();
        var deserializer = new ModelDeserializer();

        using var memoryStream = new MemoryStream();

        // Act
        serializer.Serialize(model, memoryStream);
        memoryStream.Position = 0;

        var deserializedParams = deserializer.Deserialize(memoryStream);

        // Assert
        Assert.Equal(4, deserializedParams.Count);
        Assert.True(CompareParameters(model.GetNamedParameters().ToList(), deserializedParams.ToList()));
    }

    [Fact]
    public void Deserialize_UnknownVersion_ThrowsInvalidDataException()
    {
        // Arrange
        var deserializer = new ModelDeserializer();
        using var memoryStream = new MemoryStream();

        // Write header with unknown version
        var writer = new BinaryWriter(memoryStream);
        writer.Write(FileFormatSpec.MagicBytes);
        writer.Write((ushort)999); // Unknown version
        writer.Write((ushort)0); // Flags
        writer.Write((uint)0); // HeaderChecksum
        writer.Write((uint)0); // Reserved
        writer.Write(0); // MetadataLength
        writer.Write((ulong)0); // FooterChecksum

        memoryStream.Position = 0;

        // Act & Assert
        Assert.Throws<InvalidDataException>(() => deserializer.Deserialize(memoryStream));
    }

    /// <summary>
    /// Creates a simple test model with 2 linear layers.
    /// </summary>
    private TestModel CreateSimpleModel()
    {
        return new TestModel();
    }

    /// <summary>
    /// Creates a larger test model with 4 linear layers.
    /// </summary>
    private TestLargerModel CreateLargerModel()
    {
        return new TestLargerModel();
    }

    /// <summary>
    /// Compares original and deserialized parameters for equality.
    /// </summary>
    private bool CompareParameters(
        List<(string Name, Parameter Parameter)> originalParams,
        List<KeyValuePair<string, Parameter>> deserializedParams)
    {
        if (originalParams.Count != deserializedParams.Count)
            return false;

        for (int i = 0; i < originalParams.Count; i++)
        {
            var (name, param) = originalParams[i];
            var deserializedPair = deserializedParams[i];

            if (name != deserializedPair.Key)
                return false;

            var deserializedParam = deserializedPair.Value;

            if (!param.Shape.SequenceEqual(deserializedParam.Shape))
                return false;

            if (param.Size != deserializedParam.Size)
                return false;

            for (int j = 0; j < param.Size; j++)
            {
                if (Math.Abs(param.Data[j] - deserializedParam.Data[j]) > 1e-6)
                    return false;
            }
        }

        return true;
    }
}

/// <summary>
/// Simple test model for serialization tests.
/// </summary>
internal class TestModel : Module
{
    public Parameter Weight1 { get; }
    public Parameter Bias1 { get; }

    public TestModel() : base("TestModel")
    {
        var random = new Random(42);
        var weight1Data = new float[64];
        var bias1Data = new float[8];

        for (int i = 0; i < weight1Data.Length; i++)
            weight1Data[i] = (float)random.NextDouble();

        for (int i = 0; i < bias1Data.Length; i++)
            bias1Data[i] = (float)random.NextDouble();

        Weight1 = new Parameter(weight1Data, new[] { 8, 8 }, "weight1");
        Bias1 = new Parameter(bias1Data, new[] { 8 }, "bias1");
    }

    public override Tensor Forward(Tensor input)
    {
        throw new NotImplementedException("Not used in serialization tests");
    }

    public override IEnumerable<Parameter> GetParameters()
    {
        yield return Weight1;
        yield return Bias1;
    }

    public override IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters()
    {
        yield return ("weight1", Weight1);
        yield return ("bias1", Bias1);
    }
}

/// <summary>
/// Larger test model for serialization tests.
/// </summary>
internal class TestLargerModel : Module
{
    public Parameter Weight1 { get; }
    public Parameter Bias1 { get; }
    public Parameter Weight2 { get; }
    public Parameter Bias2 { get; }

    public TestLargerModel() : base("TestLargerModel")
    {
        var random = new Random(42);
        var weight1Data = new float[256];
        var bias1Data = new float[16];
        var weight2Data = new float[64];
        var bias2Data = new float[4];

        for (int i = 0; i < weight1Data.Length; i++)
            weight1Data[i] = (float)random.NextDouble();

        for (int i = 0; i < bias1Data.Length; i++)
            bias1Data[i] = (float)random.NextDouble();

        for (int i = 0; i < weight2Data.Length; i++)
            weight2Data[i] = (float)random.NextDouble();

        for (int i = 0; i < bias2Data.Length; i++)
            bias2Data[i] = (float)random.NextDouble();

        Weight1 = new Parameter(weight1Data, new[] { 16, 16 }, "weight1");
        Bias1 = new Parameter(bias1Data, new[] { 16 }, "bias1");
        Weight2 = new Parameter(weight2Data, new[] { 4, 16 }, "weight2");
        Bias2 = new Parameter(bias2Data, new[] { 4 }, "bias2");
    }

    public override Tensor Forward(Tensor input)
    {
        throw new NotImplementedException("Not used in serialization tests");
    }

    public override IEnumerable<Parameter> GetParameters()
    {
        yield return Weight1;
        yield return Bias1;
        yield return Weight2;
        yield return Bias2;
    }

    public override IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters()
    {
        yield return ("weight1", Weight1);
        yield return ("bias1", Bias1);
        yield return ("weight2", Weight2);
        yield return ("bias2", Bias2);
    }
}
