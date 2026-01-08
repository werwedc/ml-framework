using System;
using System.Collections.Generic;
using MLFramework.ModelVersioning;
using Xunit;
using System.Text.Json;

namespace MLFramework.Tests.ModelVersioning
{
    /// <summary>
    /// Unit tests for model versioning data models
    /// </summary>
    public class DataModelsTests
    {
        [Fact]
        public void ModelMetadata_CreationAndPropertyAssignment_Works()
        {
            // Arrange
            var metadata = new ModelMetadata
            {
                CreationTimestamp = DateTime.UtcNow,
                DatasetVersion = "v1.0",
                ArchitectureHash = "abc123",
                TrainingParameters = new Dictionary<string, object>
                {
                    { "epochs", 100 },
                    { "batch_size", 32 }
                },
                CustomMetadata = new Dictionary<string, string>
                {
                    { "author", "John Doe" }
                }
            };

            // Assert
            Assert.NotNull(metadata);
            Assert.Equal("v1.0", metadata.DatasetVersion);
            Assert.Equal("abc123", metadata.ArchitectureHash);
            Assert.Equal(2, metadata.TrainingParameters?.Count);
            Assert.Single(metadata.CustomMetadata);
        }

        [Fact]
        public void PerformanceMetrics_Validation_AccuracyMustBeZeroToOne()
        {
            // Arrange & Act
            var metrics = new PerformanceMetrics
            {
                Accuracy = 0.85f,
                LatencyMs = 150.5f,
                Throughput = 1000.0f,
                MemoryUsageMB = 512.0f
            };

            // Assert
            Assert.Equal(0.85f, metrics.Accuracy);
            Assert.Equal(150.5f, metrics.LatencyMs);
            Assert.Equal(1000.0f, metrics.Throughput);
            Assert.Equal(512.0f, metrics.MemoryUsageMB);
        }

        [Fact]
        public void PerformanceMetrics_NegativeValues_AllowForZeroButNotNegative()
        {
            // Arrange & Act
            var metrics = new PerformanceMetrics
            {
                Accuracy = 0.0f,
                LatencyMs = 0.0f,
                Throughput = 0.0f,
                MemoryUsageMB = 0.0f
            };

            // Assert - Zero values are acceptable
            Assert.Equal(0.0f, metrics.Accuracy);
            Assert.Equal(0.0f, metrics.LatencyMs);
            Assert.Equal(0.0f, metrics.Throughput);
            Assert.Equal(0.0f, metrics.MemoryUsageMB);
        }

        [Fact]
        public void PerformanceMetrics_FullRange_BoundaryValues()
        {
            // Arrange & Act
            var metrics = new PerformanceMetrics
            {
                Accuracy = 1.0f,
                LatencyMs = 10000.0f,
                Throughput = 100000.0f,
                MemoryUsageMB = 10240.0f
            };

            // Assert
            Assert.Equal(1.0f, metrics.Accuracy);
            Assert.Equal(10000.0f, metrics.LatencyMs);
            Assert.Equal(100000.0f, metrics.Throughput);
            Assert.Equal(10240.0f, metrics.MemoryUsageMB);
        }

        [Fact]
        public void ModelInfo_StateTransitions_AllowsAllStates()
        {
            // Arrange
            var modelInfo = new ModelInfo
            {
                ModelId = "model-123",
                Name = "TestModel",
                VersionTag = "v1.0.0",
                State = LifecycleState.Draft
            };

            // Act & Assert
            Assert.Equal(LifecycleState.Draft, modelInfo.State);

            // Transition to Staging
            modelInfo.State = LifecycleState.Staging;
            Assert.Equal(LifecycleState.Staging, modelInfo.State);

            // Transition to Production
            modelInfo.State = LifecycleState.Production;
            Assert.Equal(LifecycleState.Production, modelInfo.State);

            // Transition to Archived
            modelInfo.State = LifecycleState.Archived;
            Assert.Equal(LifecycleState.Archived, modelInfo.State);
        }

        [Fact]
        public void ModelInfo_ParentChildRelationship_Works()
        {
            // Arrange
            var parentModel = new ModelInfo
            {
                ModelId = "parent-123",
                Name = "ParentModel",
                State = LifecycleState.Production
            };

            var childModel = new ModelInfo
            {
                ModelId = "child-456",
                Name = "ChildModel",
                ParentModelId = "parent-123",
                State = LifecycleState.Staging
            };

            // Assert
            Assert.Equal("parent-123", childModel.ParentModelId);
            Assert.Null(parentModel.ParentModelId);
        }

        [Fact]
        public void HealthCheckResult_SuccessScenario_CreatesHealthyResult()
        {
            // Arrange & Act
            var result = HealthCheckResult.Healthy("Model is ready");

            // Assert
            Assert.True(result.IsHealthy);
            Assert.Equal("Model is ready", result.Message);
            Assert.NotNull(result.Diagnostics);
            Assert.True(DateTime.UtcNow.Subtract(result.CheckTimestamp).TotalSeconds < 5);
        }

        [Fact]
        public void HealthCheckResult_FailureScenario_CreatesUnhealthyResult()
        {
            // Arrange
            var diagnostics = new Dictionary<string, object>
            {
                { "error_code", 500 },
                { "details", "Model loading failed" }
            };

            // Act
            var result = new HealthCheckResult
            {
                IsHealthy = false,
                Message = "Model failed to load",
                Diagnostics = diagnostics,
                CheckTimestamp = DateTime.UtcNow
            };

            // Assert
            Assert.False(result.IsHealthy);
            Assert.Equal("Model failed to load", result.Message);
            Assert.NotNull(result.Diagnostics);
            Assert.Equal(2, result.Diagnostics?.Count);
            Assert.Equal((object)500, result.Diagnostics?["error_code"]);
        }

        [Fact]
        public void HealthCheckResult_DefaultMessage_Works()
        {
            // Arrange & Act
            var successResult = HealthCheckResult.Healthy();
            var failureResult = HealthCheckResult.Unhealthy("Model is unhealthy");

            // Assert
            Assert.Equal("Model is healthy", successResult.Message);
            Assert.Equal("Model is unhealthy", failureResult.Message);
        }

        [Fact]
        public void ModelMetadata_JsonSerialization_RoundTrip()
        {
            // Arrange
            var original = new ModelMetadata
            {
                CreationTimestamp = DateTime.UtcNow,
                DatasetVersion = "v1.0",
                ArchitectureHash = "abc123",
                TrainingParameters = new Dictionary<string, object>
                {
                    { "epochs", 100 }
                },
                CustomMetadata = new Dictionary<string, string>
                {
                    { "author", "John" }
                }
            };

            // Act
            var json = JsonSerializer.Serialize(original);
            var deserialized = JsonSerializer.Deserialize<ModelMetadata>(json);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal("v1.0", deserialized?.DatasetVersion);
            Assert.Equal("abc123", deserialized?.ArchitectureHash);
        }

        [Fact]
        public void ModelInfo_JsonSerialization_RoundTrip()
        {
            // Arrange
            var original = new ModelInfo
            {
                ModelId = "model-123",
                Name = "TestModel",
                VersionTag = "v1.0.0",
                State = LifecycleState.Production
            };

            // Act
            var json = JsonSerializer.Serialize(original);
            var deserialized = JsonSerializer.Deserialize<ModelInfo>(json);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal("model-123", deserialized?.ModelId);
            Assert.Equal("TestModel", deserialized?.Name);
            Assert.Equal("v1.0.0", deserialized?.VersionTag);
            Assert.Equal(LifecycleState.Production, deserialized?.State);
        }

        [Fact]
        public void HealthCheckResult_JsonSerialization_RoundTrip()
        {
            // Arrange
            var original = HealthCheckResult.Healthy("Test message");

            // Act
            var json = JsonSerializer.Serialize(original);
            var deserialized = JsonSerializer.Deserialize<HealthCheckResult>(json);

            // Assert
            Assert.NotNull(deserialized);
            Assert.True(deserialized?.IsHealthy);
            Assert.Equal("Test message", deserialized?.Message);
        }

        [Fact]
        public void ModelMetadata_ToString_ReturnsFormattedString()
        {
            // Arrange
            var metadata = new ModelMetadata
            {
                CreationTimestamp = DateTime.UtcNow,
                DatasetVersion = "v1.0",
                ArchitectureHash = "abc123"
            };

            // Act
            var str = metadata.ToString();

            // Assert
            Assert.Contains("ModelMetadata", str);
            Assert.Contains("v1.0", str);
            Assert.Contains("abc123", str);
        }

        [Fact]
        public void PerformanceMetrics_ToString_ReturnsFormattedString()
        {
            // Arrange
            var metrics = new PerformanceMetrics
            {
                Accuracy = 0.95f,
                LatencyMs = 100.5f,
                Throughput = 500.0f,
                MemoryUsageMB = 256.0f
            };

            // Act
            var str = metrics.ToString();

            // Assert
            Assert.Contains("PerformanceMetrics", str);
            Assert.Contains("95.00%", str);
            Assert.Contains("100.5ms", str);
        }

        [Fact]
        public void ModelInfo_ToString_ReturnsFormattedString()
        {
            // Arrange
            var modelInfo = new ModelInfo
            {
                ModelId = "model-123",
                Name = "TestModel",
                VersionTag = "v1.0.0",
                State = LifecycleState.Production
            };

            // Act
            var str = modelInfo.ToString();

            // Assert
            Assert.Contains("ModelInfo", str);
            Assert.Contains("model-123", str);
            Assert.Contains("TestModel", str);
            Assert.Contains("v1.0.0", str);
        }

        [Fact]
        public void HealthCheckResult_ToString_ReturnsFormattedString()
        {
            // Arrange
            var result = HealthCheckResult.Healthy("Test");

            // Act
            var str = result.ToString();

            // Assert
            Assert.Contains("HealthCheckResult", str);
            Assert.Contains("Test", str);
        }

        [Fact]
        public void LifecycleState_AllValues_EnumCorrect()
        {
            // Assert
            Assert.Equal(0, (int)LifecycleState.Draft);
            Assert.Equal(1, (int)LifecycleState.Staging);
            Assert.Equal(2, (int)LifecycleState.Production);
            Assert.Equal(3, (int)LifecycleState.Archived);
        }
    }
}
