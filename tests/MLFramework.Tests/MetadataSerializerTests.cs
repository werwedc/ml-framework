using MLFramework.ModelRegistry;
using System.Text.Json;

namespace MLFramework.Tests.Serving.Deployment;

[TestFixture]
public class MetadataSerializerTests
{
    private MetadataSerializer _serializer;

    [SetUp]
    public void SetUp()
    {
        _serializer = new MetadataSerializer();
    }

    [Test]
    public void Serialize_SimpleMetadata_ReturnsValidJson()
    {
        // Arrange
        var metadata = new ModelMetadata
        {
            Version = "1.0.0",
            TrainingDate = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc),
            ArtifactPath = "/models/model1.bin"
        };

        // Act
        var json = _serializer.Serialize(metadata);

        // Assert
        Assert.That(json, Is.Not.Null);
        Assert.That(json, Does.Contain("\"version\": \"1.0.0\""));
        Assert.That(json, Does.Contain("\"artifactPath\": \"/models/model1.bin\""));
        Assert.That(json, Does.Contain("\"trainingDate\""));

        // Verify it's valid JSON
        Assert.DoesNotThrow(() => JsonDocument.Parse(json));
    }

    [Test]
    public void Deserialize_ValidJson_RestoresAllFields()
    {
        // Arrange
        var json = @"{
            ""version"": ""1.2.3"",
            ""trainingDate"": ""2024-06-15T10:30:00Z"",
            ""artifactPath"": ""/models/model2.bin"",
            ""hyperparameters"": {
                ""learningRate"": 0.001,
                ""batchSize"": 32
            },
            ""performanceMetrics"": {
                ""accuracy"": 0.95,
                ""f1Score"": 0.93
            }
        }";

        // Act
        var metadata = _serializer.Deserialize(json);

        // Assert
        Assert.That(metadata, Is.Not.Null);
        Assert.That(metadata.Version, Is.EqualTo("1.2.3"));
        Assert.That(metadata.ArtifactPath, Is.EqualTo("/models/model2.bin"));
        Assert.That(metadata.TrainingDate, Is.EqualTo(new DateTime(2024, 6, 15, 10, 30, 0, DateTimeKind.Utc)));
        Assert.That(metadata.Hyperparameters.Count, Is.EqualTo(2));
        Assert.That(metadata.Hyperparameters["learningRate"], Is.EqualTo(0.001));
        Assert.That(metadata.Hyperparameters["batchSize"], Is.EqualTo(32));
        Assert.That(metadata.PerformanceMetrics.Count, Is.EqualTo(2));
        Assert.That(metadata.PerformanceMetrics["accuracy"], Is.EqualTo(0.95f).Within(0.0001f));
        Assert.That(metadata.PerformanceMetrics["f1Score"], Is.EqualTo(0.93f).Within(0.0001f));
    }

    [Test]
    public void Serialize_WithNestedDictionaries_PreservesStructure()
    {
        // Arrange
        var metadata = new ModelMetadata
        {
            Version = "2.0.0",
            TrainingDate = DateTime.UtcNow,
            ArtifactPath = "/models/model3.bin",
            Hyperparameters = new Dictionary<string, object>
            {
                { "optimizer", "adam" },
                { "learningRate", 0.0001 },
                { "layers", new Dictionary<string, object>
                    {
                        { "hidden1", 256 },
                        { "hidden2", 128 }
                    }
                }
            }
        };

        // Act
        var json = _serializer.Serialize(metadata);

        // Assert
        var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        Assert.That(root.GetProperty("version").GetString(), Is.EqualTo("2.0.0"));
        Assert.That(root.GetProperty("hyperparameters").GetProperty("optimizer").GetString(), Is.EqualTo("adam"));
        Assert.That(root.GetProperty("hyperparameters").GetProperty("layers").GetProperty("hidden1").GetInt32(), Is.EqualTo(256));
        Assert.That(root.GetProperty("hyperparameters").GetProperty("layers").GetProperty("hidden2").GetInt32(), Is.EqualTo(128));
    }

    [Test]
    public void Serialize_WithVariousValueTypes_HandlesCorrectly()
    {
        // Arrange
        var metadata = new ModelMetadata
        {
            Version = "1.0.0",
            TrainingDate = DateTime.UtcNow,
            ArtifactPath = "/models/model.bin",
            Hyperparameters = new Dictionary<string, object>
            {
                { "intValue", 42 },
                { "longValue", 1234567890L },
                { "doubleValue", 3.14159 },
                { "boolValue", true },
                { "stringValue", "test" }
            }
        };

        // Act
        var json = _serializer.Serialize(metadata);

        // Assert
        var doc = JsonDocument.Parse(json);
        var hp = doc.RootElement.GetProperty("hyperparameters");

        Assert.That(hp.GetProperty("intValue").GetInt32(), Is.EqualTo(42));
        Assert.That(hp.GetProperty("longValue").GetInt64(), Is.EqualTo(1234567890L));
        Assert.That(hp.GetProperty("doubleValue").GetDouble(), Is.EqualTo(3.14159).Within(0.00001));
        Assert.That(hp.GetProperty("boolValue").GetBoolean(), Is.True);
        Assert.That(hp.GetProperty("stringValue").GetString(), Is.EqualTo("test"));
    }

    [Test]
    public void Deserialize_WithMissingRequiredVersion_Throws()
    {
        // Arrange
        var json = @"{
            ""trainingDate"": ""2024-01-01T00:00:00Z"",
            ""artifactPath"": ""/models/model.bin""
        }";

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => _serializer.Deserialize(json));
    }

    [Test]
    public void Deserialize_WithMissingRequiredArtifactPath_Throws()
    {
        // Arrange
        var json = @"{
            ""version"": ""1.0.0"",
            ""trainingDate"": ""2024-01-01T00:00:00Z""
        }";

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => _serializer.Deserialize(json));
    }

    [Test]
    public void RoundTrip_SerializeDeserialize_PreservesAllData()
    {
        // Arrange
        var original = new ModelMetadata
        {
            Version = "1.5.2",
            TrainingDate = new DateTime(2024, 3, 15, 14, 30, 45, DateTimeKind.Utc),
            ArtifactPath = "/models/complex_model.bin",
            Hyperparameters = new Dictionary<string, object>
            {
                { "learningRate", 0.01 },
                { "epochs", 100 },
                { "dropout", 0.5 },
                { "useBatchNorm", true }
            },
            PerformanceMetrics = new Dictionary<string, float>
            {
                { "accuracy", 0.92f },
                { "precision", 0.90f },
                { "recall", 0.94f },
                { "f1Score", 0.92f }
            }
        };

        // Act
        var json = _serializer.Serialize(original);
        var deserialized = _serializer.Deserialize(json);

        // Assert
        Assert.That(deserialized.Version, Is.EqualTo(original.Version));
        Assert.That(deserialized.TrainingDate, Is.EqualTo(original.TrainingDate));
        Assert.That(deserialized.ArtifactPath, Is.EqualTo(original.ArtifactPath));

        Assert.That(deserialized.Hyperparameters.Count, Is.EqualTo(original.Hyperparameters.Count));
        foreach (var kvp in original.Hyperparameters)
        {
            Assert.That(deserialized.Hyperparameters.ContainsKey(kvp.Key), Is.True);
            Assert.That(deserialized.Hyperparameters[kvp.Key], Is.EqualTo(kvp.Value));
        }

        Assert.That(deserialized.PerformanceMetrics.Count, Is.EqualTo(original.PerformanceMetrics.Count));
        foreach (var kvp in original.PerformanceMetrics)
        {
            Assert.That(deserialized.PerformanceMetrics.ContainsKey(kvp.Key), Is.True);
            Assert.That(deserialized.PerformanceMetrics[kvp.Key], Is.EqualTo(kvp.Value).Within(0.0001f));
        }
    }

    [Test]
    public async Task SerializeAsync_SimpleMetadata_ReturnsValidJson()
    {
        // Arrange
        var metadata = new ModelMetadata
        {
            Version = "1.0.0",
            TrainingDate = DateTime.UtcNow,
            ArtifactPath = "/models/model.bin"
        };

        // Act
        var json = await _serializer.SerializeAsync(metadata);

        // Assert
        Assert.That(json, Is.Not.Null);
        Assert.That(json, Does.Contain("\"version\": \"1.0.0\""));
        Assert.DoesNotThrow(() => JsonDocument.Parse(json));
    }

    [Test]
    public async Task DeserializeAsync_ValidJson_RestoresAllFields()
    {
        // Arrange
        var json = @"{
            ""version"": ""2.1.0"",
            ""trainingDate"": ""2024-12-25T12:00:00Z"",
            ""artifactPath"": ""/models/holiday_model.bin""
        }";

        // Act
        var metadata = await _serializer.DeserializeAsync(json);

        // Assert
        Assert.That(metadata, Is.Not.Null);
        Assert.That(metadata.Version, Is.EqualTo("2.1.0"));
        Assert.That(metadata.ArtifactPath, Is.EqualTo("/models/holiday_model.bin"));
    }

    [Test]
    public void Serialize_NullMetadata_Throws()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => _serializer.Serialize(null));
    }

    [Test]
    public void Deserialize_NullJson_Throws()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _serializer.Deserialize(null));
    }

    [Test]
    public void Deserialize_EmptyJson_Throws()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _serializer.Deserialize(""));
    }

    [Test]
    public void Deserialize_InvalidJson_Throws()
    {
        // Arrange
        var invalidJson = "{ invalid json }";

        // Act & Assert
        Assert.Throws<JsonException>(() => _serializer.Deserialize(invalidJson));
    }

    [Test]
    public void SaveToFile_ValidMetadata_CreatesFile()
    {
        // Arrange
        var metadata = new ModelMetadata
        {
            Version = "1.0.0",
            TrainingDate = DateTime.UtcNow,
            ArtifactPath = "/models/model.bin"
        };
        var tempFile = Path.GetTempFileName();

        try
        {
            // Act
            _serializer.SaveToFile(tempFile, metadata);

            // Assert
            Assert.That(File.Exists(tempFile), Is.True);

            var content = File.ReadAllText(tempFile);
            Assert.That(content, Does.Contain("\"version\": \"1.0.0\""));
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void LoadFromFile_ValidFile_ReturnsMetadata()
    {
        // Arrange
        var json = @"{
            ""version"": ""1.0.0"",
            ""trainingDate"": ""2024-01-01T00:00:00Z"",
            ""artifactPath"": ""/models/model.bin""
        }";
        var tempFile = Path.GetTempFileName();

        try
        {
            File.WriteAllText(tempFile, json);

            // Act
            var metadata = _serializer.LoadFromFile(tempFile);

            // Assert
            Assert.That(metadata, Is.Not.Null);
            Assert.That(metadata.Version, Is.EqualTo("1.0.0"));
            Assert.That(metadata.ArtifactPath, Is.EqualTo("/models/model.bin"));
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void LoadFromFile_NonExistentFile_Throws()
    {
        // Arrange
        var nonExistentFile = "/tmp/non_existent_file_12345.json";

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() => _serializer.LoadFromFile(nonExistentFile));
    }

    [Test]
    public void SaveToFile_NullMetadata_Throws()
    {
        // Arrange
        var tempFile = Path.GetTempFileName();

        try
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => _serializer.SaveToFile(tempFile, null));
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void Serialize_WithEmptyVersion_Throws()
    {
        // Arrange
        var metadata = new ModelMetadata
        {
            Version = "",
            ArtifactPath = "/models/model.bin"
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => _serializer.Serialize(metadata));
    }

    [Test]
    public void Serialize_WithEmptyArtifactPath_Throws()
    {
        // Arrange
        var metadata = new ModelMetadata
        {
            Version = "1.0.0",
            ArtifactPath = ""
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => _serializer.Serialize(metadata));
    }
}
