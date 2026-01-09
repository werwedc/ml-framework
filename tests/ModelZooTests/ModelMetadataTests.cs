using System;
using System.IO;
using System.Collections.Generic;
using Xunit;
using MLFramework.ModelZoo;

namespace MLFramework.Tests.ModelZooTests
{
    /// <summary>
    /// Unit tests for ModelMetadata and ModelMetadataValidator.
    /// </summary>
    public class ModelMetadataTests
    {
        /// <summary>
        /// Creates a valid ModelMetadata object for testing.
        /// </summary>
        private ModelMetadata CreateValidMetadata()
        {
            return new ModelMetadata
            {
                Name = "resnet50",
                Version = "1.0.0",
                Architecture = "ResNet",
                Variants = new[] { "resnet18", "resnet50", "resnet101" },
                PretrainedOn = "ImageNet",
                PerformanceMetrics = new Dictionary<string, double>
                {
                    { "top1", 0.761 },
                    { "top5", 0.930 }
                },
                InputShape = new[] { 3, 224, 224 },
                OutputShape = new[] { 1000 },
                NumParameters = 25600000,
                FileSizeBytes = 102400000,
                License = "Apache-2.0",
                PaperUrl = "https://arxiv.org/abs/1512.03385",
                SourceCodeUrl = "https://github.com/pytorch/vision",
                Sha256Checksum = new string('0', 64), // Valid 64-char hex string
                DownloadUrl = "https://example.com/resnet50.bin",
                MirrorUrls = new[] { "https://mirror1.example.com/resnet50.bin" }
            };
        }

        [Fact]
        public void ValidMetadata_PassesValidation()
        {
            // Arrange
            var metadata = CreateValidMetadata();

            // Act
            var errors = ModelMetadataValidator.Validate(metadata);

            // Assert
            Assert.Empty(errors);
            Assert.True(ModelMetadataValidator.IsValid(metadata));
        }

        [Fact]
        public void NullMetadata_ReturnsValidationError()
        {
            // Act
            var errors = ModelMetadataValidator.Validate(null);

            // Assert
            Assert.Single(errors);
            Assert.Contains("cannot be null", errors[0]);
            Assert.False(ModelMetadataValidator.IsValid(null));
        }

        [Fact]
        public void MissingRequiredFields_ReturnsValidationErrors()
        {
            // Arrange
            var metadata = new ModelMetadata();

            // Act
            var errors = ModelMetadataValidator.Validate(metadata);

            // Assert
            Assert.NotEmpty(errors);
            Assert.Contains(errors, e => e.Contains("Name") || e.ToLower().Contains("name"));
            Assert.Contains(errors, e => e.Contains("Version") || e.ToLower().Contains("version"));
            Assert.Contains(errors, e => e.Contains("Architecture") || e.ToLower().Contains("architecture"));
            Assert.Contains(errors, e => e.Contains("License") || e.ToLower().Contains("license"));
        }

        [Fact]
        public void InvalidSemanticVersion_ReturnsValidationError()
        {
            // Arrange
            var metadata = CreateValidMetadata();
            metadata.Version = "1.0"; // Missing patch version

            // Act
            var errors = ModelMetadataValidator.Validate(metadata);

            // Assert
            Assert.Contains(errors, e => e.Contains("semantic versioning"));
        }

        [Fact]
        public void ValidSemanticVersions_Allowed()
        {
            // Arrange
            var validVersions = new[] { "1.0.0", "2.1.3", "0.0.1", "10.20.30", "1.0.0-beta", "1.0.0-alpha.1", "1.0.0+build.1" };

            foreach (var version in validVersions)
            {
                var metadata = CreateValidMetadata();
                metadata.Version = version;

                // Act
                var errors = ModelMetadataValidator.Validate(metadata);

                // Assert
                Assert.DoesNotContain(errors, e => e.Contains("semantic versioning"));
            }
        }

        [Fact]
        public void InvalidUrl_ReturnsValidationError()
        {
            // Arrange
            var metadata = CreateValidMetadata();
            metadata.DownloadUrl = "not-a-url";

            // Act
            var errors = ModelMetadataValidator.Validate(metadata);

            // Assert
            Assert.Contains(errors, e => e.Contains("Download URL") && e.Contains("valid URL"));
        }

        [Fact]
        public void ValidUrls_Accepted()
        {
            // Arrange
            var metadata = CreateValidMetadata();
            metadata.DownloadUrl = "https://example.com/model.bin";
            metadata.PaperUrl = "https://arxiv.org/abs/1234.5678";
            metadata.SourceCodeUrl = "https://github.com/example/repo";

            // Act
            var errors = ModelMetadataValidator.Validate(metadata);

            // Assert
            Assert.DoesNotContain(errors, e => e.Contains("valid URL"));
        }

        [Fact]
        public void InvalidSha256Checksum_ReturnsValidationError()
        {
            // Arrange
            var metadata = CreateValidMetadata();
            metadata.Sha256Checksum = "invalid-checksum";

            // Act
            var errors = ModelMetadataValidator.Validate(metadata);

            // Assert
            Assert.Contains(errors, e => e.Contains("SHA256") && e.Contains("64-character"));
        }

        [Fact]
        public void ValidSha256Checksum_Accepted()
        {
            // Arrange
            var metadata = CreateValidMetadata();
            metadata.Sha256Checksum = "a1b2c3d4e5f67890123456789012345678901234567890123456789012345678";

            // Act
            var errors = ModelMetadataValidator.Validate(metadata);

            // Assert
            Assert.DoesNotContain(errors, e => e.Contains("SHA256"));
        }

        [Fact]
        public void PerformanceMetricOutOfRange_ReturnsValidationError()
        {
            // Arrange
            var metadata = CreateValidMetadata();
            metadata.PerformanceMetrics["top1"] = 1.5; // Should be 0-1

            // Act
            var errors = ModelMetadataValidator.Validate(metadata);

            // Assert
            Assert.Contains(errors, e => e.Contains("top1") && e.Contains("[0, 1]"));
        }

        [Fact]
        public void NegativePerformanceMetric_ReturnsValidationError()
        {
            // Arrange
            var metadata = CreateValidMetadata();
            metadata.PerformanceMetrics["custom_metric"] = -0.5;

            // Act
            var errors = ModelMetadataValidator.Validate(metadata);

            // Assert
            Assert.Contains(errors, e => e.Contains("negative value"));
        }

        [Fact]
        public void NegativeNumericFields_ReturnValidationErrors()
        {
            // Arrange
            var metadata = CreateValidMetadata();
            metadata.NumParameters = -100;
            metadata.FileSizeBytes = -500;

            // Act
            var errors = ModelMetadataValidator.Validate(metadata);

            // Assert
            Assert.Contains(errors, e => e.Contains("negative"));
        }

        [Fact]
        public void InvalidInputShape_ReturnsValidationError()
        {
            // Arrange
            var metadata = CreateValidMetadata();
            metadata.InputShape = Array.Empty<int>();

            // Act
            var errors = ModelMetadataValidator.Validate(metadata);

            // Assert
            Assert.Contains(errors, e => e.Contains("Input shape") && e.ToLower().Contains("empty"));
        }

        [Fact]
        public void InvalidShapeDimensions_ReturnsValidationError()
        {
            // Arrange
            var metadata = CreateValidMetadata();
            metadata.InputShape = new[] { 3, -1, 224 };

            // Act
            var errors = ModelMetadataValidator.Validate(metadata);

            // Assert
            Assert.Contains(errors, e => e.Contains("invalid dimension"));
        }

        [Fact]
        public void ValidateOrThrow_ThrowsOnInvalidMetadata()
        {
            // Arrange
            var metadata = new ModelMetadata();

            // Act & Assert
            Assert.Throws<ValidationException>(() => ModelMetadataValidator.ValidateOrThrow(metadata));
        }

        [Fact]
        public void ValidateOrThrow_DoesNotThrowOnValidMetadata()
        {
            // Arrange
            var metadata = CreateValidMetadata();

            // Act & Assert
            var exception = Record.Exception(() => ModelMetadataValidator.ValidateOrThrow(metadata));
            Assert.Null(exception);
        }

        [Fact]
        public void ToJson_SerializesCorrectly()
        {
            // Arrange
            var metadata = CreateValidMetadata();

            // Act
            var json = metadata.ToJson();

            // Assert
            Assert.Contains("\"name\"", json);
            Assert.Contains("\"resnet50\"", json);
            Assert.Contains("\"version\"", json);
            Assert.Contains("\"1.0.0\"", json);
        }

        [Fact]
        public void FromJson_DeserializesCorrectly()
        {
            // Arrange
            var originalMetadata = CreateValidMetadata();
            var json = originalMetadata.ToJson();

            // Act
            var deserializedMetadata = ModelMetadata.FromJson(json);

            // Assert
            Assert.Equal(originalMetadata.Name, deserializedMetadata.Name);
            Assert.Equal(originalMetadata.Version, deserializedMetadata.Version);
            Assert.Equal(originalMetadata.Architecture, deserializedMetadata.Architecture);
            Assert.Equal(originalMetadata.License, deserializedMetadata.License);
            Assert.Equal(originalMetadata.Sha256Checksum, deserializedMetadata.Sha256Checksum);
        }

        [Fact]
        public void SerializationRoundTrip_PreservesAllData()
        {
            // Arrange
            var originalMetadata = CreateValidMetadata();

            // Act
            var json = originalMetadata.ToJson();
            var deserializedMetadata = ModelMetadata.FromJson(json);

            // Assert
            Assert.Equal(originalMetadata.Name, deserializedMetadata.Name);
            Assert.Equal(originalMetadata.Version, deserializedMetadata.Version);
            Assert.Equal(originalMetadata.Architecture, deserializedMetadata.Architecture);
            Assert.Equal(originalMetadata.Variants, deserializedMetadata.Variants);
            Assert.Equal(originalMetadata.PretrainedOn, deserializedMetadata.PretrainedOn);
            Assert.Equal(originalMetadata.InputShape, deserializedMetadata.InputShape);
            Assert.Equal(originalMetadata.OutputShape, deserializedMetadata.OutputShape);
            Assert.Equal(originalMetadata.NumParameters, deserializedMetadata.NumParameters);
            Assert.Equal(originalMetadata.FileSizeBytes, deserializedMetadata.FileSizeBytes);
            Assert.Equal(originalMetadata.License, deserializedMetadata.License);
            Assert.Equal(originalMetadata.PaperUrl, deserializedMetadata.PaperUrl);
            Assert.Equal(originalMetadata.SourceCodeUrl, deserializedMetadata.SourceCodeUrl);
            Assert.Equal(originalMetadata.Sha256Checksum, deserializedMetadata.Sha256Checksum);
            Assert.Equal(originalMetadata.DownloadUrl, deserializedMetadata.DownloadUrl);
            Assert.Equal(originalMetadata.MirrorUrls, deserializedMetadata.MirrorUrls);
        }

        [Fact]
        public void SaveToJsonFile_CreatesFile()
        {
            // Arrange
            var metadata = CreateValidMetadata();
            var tempFile = Path.GetTempFileName();

            try
            {
                // Act
                metadata.SaveToJsonFile(tempFile);

                // Assert
                Assert.True(File.Exists(tempFile));
            }
            finally
            {
                // Cleanup
                if (File.Exists(tempFile))
                {
                    File.Delete(tempFile);
                }
            }
        }

        [Fact]
        public void LoadFromJsonFile_LoadsCorrectly()
        {
            // Arrange
            var originalMetadata = CreateValidMetadata();
            var tempFile = Path.GetTempFileName();

            try
            {
                // Act
                originalMetadata.SaveToJsonFile(tempFile);
                var loadedMetadata = ModelMetadata.LoadFromJsonFile(tempFile);

                // Assert
                Assert.Equal(originalMetadata.Name, loadedMetadata.Name);
                Assert.Equal(originalMetadata.Version, loadedMetadata.Version);
            }
            finally
            {
                // Cleanup
                if (File.Exists(tempFile))
                {
                    File.Delete(tempFile);
                }
            }
        }

        [Fact]
        public void EmptyArrays_HandledCorrectly()
        {
            // Arrange
            var metadata = new ModelMetadata
            {
                Name = "test",
                Version = "1.0.0",
                Architecture = "TestArch",
                PretrainedOn = "TestDataset",
                InputShape = new[] { 1, 28, 28 },
                License = "MIT",
                Sha256Checksum = new string('0', 64),
                DownloadUrl = "https://example.com/test.bin",
                Variants = Array.Empty<string>(),
                PerformanceMetrics = new Dictionary<string, double>(),
                MirrorUrls = Array.Empty<string>()
            };

            // Act
            var errors = ModelMetadataValidator.Validate(metadata);

            // Assert
            Assert.Empty(errors);
        }

        [Fact]
        public void NullOptionalFields_HandledCorrectly()
        {
            // Arrange
            var metadata = CreateValidMetadata();
            metadata.PaperUrl = null;
            metadata.SourceCodeUrl = null;

            // Act
            var errors = ModelMetadataValidator.Validate(metadata);

            // Assert
            Assert.Empty(errors);
        }

        [Fact]
        public void CaseInsensitiveUrlValidation_WorksCorrectly()
        {
            // Arrange
            var metadata = CreateValidMetadata();
            metadata.DownloadUrl = "HTTPS://EXAMPLE.COM/MODEL.BIN";

            // Act
            var errors = ModelMetadataValidator.Validate(metadata);

            // Assert
            Assert.DoesNotContain(errors, e => e.Contains("valid URL"));
        }
    }
}
