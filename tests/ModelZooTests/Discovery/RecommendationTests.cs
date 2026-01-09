using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.ModelZoo;
using MLFramework.ModelZoo.Discovery;
using Xunit;

namespace ModelZooTests.Discovery
{
    /// <summary>
    /// Comprehensive unit tests for ModelRecommendationEngine.
    /// </summary>
    public class RecommendationTests : IDisposable
    {
        private readonly ModelRegistry _registry;
        private readonly ModelRecommendationEngine _engine;
        private readonly LatencyEstimator _latencyEstimator;
        private readonly MemoryEstimator _memoryEstimator;

        public RecommendationTests()
        {
            _registry = new ModelRegistry();
            _engine = new ModelRecommendationEngine(_registry);
            _latencyEstimator = new LatencyEstimator();
            _memoryEstimator = new MemoryEstimator();
            SetupTestModels();
        }

        private void SetupTestModels()
        {
            // ResNet-50 (Large, high accuracy)
            var resnet50 = new ModelMetadata
            {
                Name = "resnet50",
                Version = "1.0.0",
                Architecture = "ResNet",
                Task = TaskType.ImageClassification,
                PretrainedOn = "ImageNet",
                PerformanceMetrics = new Dictionary<string, double>
                {
                    { "accuracy", 0.76 },
                    { "top1", 0.76 },
                    { "top5", 0.93 }
                },
                InputShape = new[] { 3, 224, 224 },
                OutputShape = new[] { 1000 },
                NumParameters = 25_600_000,
                FileSizeBytes = 98 * 1024 * 1024,
                License = "MIT",
                Sha256Checksum = "abc123",
                DownloadUrl = "https://example.com/resnet50.pt"
            };

            // ResNet-18 (Smaller, faster)
            var resnet18 = new ModelMetadata
            {
                Name = "resnet18",
                Version = "1.0.0",
                Architecture = "ResNet",
                Task = TaskType.ImageClassification,
                PretrainedOn = "ImageNet",
                PerformanceMetrics = new Dictionary<string, double>
                {
                    { "accuracy", 0.70 },
                    { "top1", 0.70 },
                    { "top5", 0.89 }
                },
                InputShape = new[] { 3, 224, 224 },
                OutputShape = new[] { 1000 },
                NumParameters = 11_700_000,
                FileSizeBytes = 45 * 1024 * 1024,
                License = "MIT",
                Sha256Checksum = "def456",
                DownloadUrl = "https://example.com/resnet18.pt"
            };

            // MobileNetV2 (Very small, edge-optimized)
            var mobilenet = new ModelMetadata
            {
                Name = "mobilenet_v2",
                Version = "1.0.0",
                Architecture = "MobileNetV2",
                Task = TaskType.ImageClassification,
                PretrainedOn = "ImageNet",
                PerformanceMetrics = new Dictionary<string, double>
                {
                    { "accuracy", 0.72 },
                    { "top1", 0.72 },
                    { "top5", 0.91 }
                },
                InputShape = new[] { 3, 224, 224 },
                OutputShape = new[] { 1000 },
                NumParameters = 3_400_000,
                FileSizeBytes = 14 * 1024 * 1024,
                License = "Apache-2.0",
                Sha256Checksum = "ghi789",
                DownloadUrl = "https://example.com/mobilenet.pt"
            };

            // BERT-Base (Text classification)
            var bertBase = new ModelMetadata
            {
                Name = "bert_base_uncased",
                Version = "1.0.0",
                Architecture = "BERT",
                Task = TaskType.TextClassification,
                PretrainedOn = "BookCorpus+Wiki",
                PerformanceMetrics = new Dictionary<string, double>
                {
                    { "accuracy", 0.82 }
                },
                InputShape = new[] { 512 },
                OutputShape = new[] { 2 },
                NumParameters = 110_000_000,
                FileSizeBytes = 440 * 1024 * 1024,
                License = "Apache-2.0",
                Sha256Checksum = "jkl012",
                DownloadUrl = "https://example.com/bert_base.pt"
            };

            // BERT-Large (Text classification, larger)
            var bertLarge = new ModelMetadata
            {
                Name = "bert_large_uncased",
                Version = "1.0.0",
                Architecture = "BERT",
                Task = TaskType.TextClassification,
                PretrainedOn = "BookCorpus+Wiki",
                PerformanceMetrics = new Dictionary<string, double>
                {
                    { "accuracy", 0.85 }
                },
                InputShape = new[] { 512 },
                OutputShape = new[] { 2 },
                NumParameters = 340_000_000,
                FileSizeBytes = 1360 * 1024 * 1024,
                License = "Apache-2.0",
                Sha256Checksum = "mno345",
                DownloadUrl = "https://example.com/bert_large.pt"
            };

            // EfficientNet-B0 (Balanced CNN)
            var efficientnet = new ModelMetadata
            {
                Name = "efficientnet_b0",
                Version = "1.0.0",
                Architecture = "EfficientNet",
                Task = TaskType.ImageClassification,
                PretrainedOn = "ImageNet",
                PerformanceMetrics = new Dictionary<string, double>
                {
                    { "accuracy", 0.77 },
                    { "top1", 0.77 },
                    { "top5", 0.93 }
                },
                InputShape = new[] { 3, 224, 224 },
                OutputShape = new[] { 1000 },
                NumParameters = 5_300_000,
                FileSizeBytes = 20 * 1024 * 1024,
                License = "Apache-2.0",
                Sha256Checksum = "pqr678",
                DownloadUrl = "https://example.com/efficientnet.pt"
            };

            _registry.Register(resnet50);
            _registry.Register(resnet18);
            _registry.Register(mobilenet);
            _registry.Register(bertBase);
            _registry.Register(bertLarge);
            _registry.Register(efficientnet);
        }

        [Fact]
        public void RecommendFor_ImageClassification_ReturnsModel()
        {
            // Arrange
            var inputShape = new Shape(new[] { 3, 224, 224 });

            // Act
            var recommendation = _engine.RecommendFor(inputShape, TaskType.ImageClassification);

            // Assert
            Assert.NotNull(recommendation);
            Assert.NotNull(recommendation.Model);
            Assert.Equal(TaskType.ImageClassification, recommendation.Model.Task);
            Assert.True(recommendation.CompatibilityScore > 0);
        }

        [Fact]
        public void RecommendFor_TextClassification_ReturnsModel()
        {
            // Arrange
            var inputShape = new Shape(new[] { 512 });

            // Act
            var recommendation = _engine.RecommendFor(inputShape, TaskType.TextClassification);

            // Assert
            Assert.NotNull(recommendation);
            Assert.NotNull(recommendation.Model);
            Assert.Equal(TaskType.TextClassification, recommendation.Model.Task);
            Assert.True(recommendation.CompatibilityScore > 0);
        }

        [Fact]
        public void RecommendFor_WithLatencyConstraint_RespectsConstraint()
        {
            // Arrange
            var inputShape = new Shape(new[] { 3, 224, 224 });
            var constraints = ModelConstraints.Create()
                .WithInputShape(inputShape)
                .WithTask(TaskType.ImageClassification)
                .WithMaxLatency(10.0f);

            // Act
            var recommendation = _engine.RecommendFor(inputShape, TaskType.ImageClassification, constraints);

            // Assert
            Assert.NotNull(recommendation);
            Assert.True(recommendation.EstimatedLatency <= constraints.MaxLatency.Value,
                $"Estimated latency {recommendation.EstimatedLatency} exceeds constraint {constraints.MaxLatency}");
        }

        [Fact]
        public void RecommendFor_WithMemoryConstraint_RespectsConstraint()
        {
            // Arrange
            var inputShape = new Shape(new[] { 3, 224, 224 });
            var maxMemory = 50 * 1024 * 1024; // 50 MB
            var constraints = ModelConstraints.Create()
                .WithInputShape(inputShape)
                .WithTask(TaskType.ImageClassification)
                .WithMaxMemory(maxMemory);

            // Act
            var recommendation = _engine.RecommendFor(inputShape, TaskType.ImageClassification, constraints);

            // Assert
            Assert.NotNull(recommendation);
            Assert.True(recommendation.EstimatedMemory <= maxMemory,
                $"Estimated memory {recommendation.EstimatedMemory} exceeds constraint {maxMemory}");
        }

        [Fact]
        public void RecommendFor_WithAccuracyConstraint_PrioritizesAccuracy()
        {
            // Arrange
            var inputShape = new Shape(new[] { 3, 224, 224 });
            var constraints = ModelConstraints.Create()
                .WithInputShape(inputShape)
                .WithTask(TaskType.ImageClassification)
                .WithMinAccuracy(0.75f);

            // Act
            var recommendation = _engine.RecommendFor(inputShape, TaskType.ImageClassification, constraints);

            // Assert
            Assert.NotNull(recommendation);
            Assert.True(recommendation.Model.PerformanceMetrics.ContainsKey("accuracy"));
            Assert.True(recommendation.Model.PerformanceMetrics["accuracy"] >= constraints.MinAccuracy.Value,
                $"Model accuracy {recommendation.Model.PerformanceMetrics["accuracy"]} below constraint {constraints.MinAccuracy}");
        }

        [Fact]
        public void RecommendMultiple_ReturnsTopNModels()
        {
            // Arrange
            var inputShape = new Shape(new[] { 3, 224, 224 });
            int topN = 3;

            // Act
            var recommendations = _engine.RecommendMultiple(inputShape, TaskType.ImageClassification, topN);

            // Assert
            Assert.NotNull(recommendations);
            Assert.Equal(topN, recommendations.Count);
            Assert.All(recommendations, r =>
            {
                Assert.NotNull(r.Model);
                Assert.Equal(TaskType.ImageClassification, r.Model.Task);
            });
        }

        [Fact]
        public void RecommendMultiple_ReturnsModelsInDescendingScoreOrder()
        {
            // Arrange
            var inputShape = new Shape(new[] { 3, 224, 224 });
            int topN = 3;

            // Act
            var recommendations = _engine.RecommendMultiple(inputShape, TaskType.ImageClassification, topN);

            // Assert
            for (int i = 1; i < recommendations.Count; i++)
            {
                Assert.True(recommendations[i - 1].CompatibilityScore >= recommendations[i].CompatibilityScore,
                    "Models should be ordered by compatibility score (descending)");
            }
        }

        [Fact]
        public void GetAlternatives_ReturnsSimilarModels()
        {
            // Arrange
            string modelName = "resnet50";
            int count = 2;

            // Act
            var alternatives = _engine.GetAlternatives(modelName, count);

            // Assert
            Assert.NotNull(alternatives);
            Assert.True(alternatives.Count <= count);
            Assert.All(alternatives, a => Assert.NotEqual(modelName, a.Name));
        }

        [Fact]
        public void GetAlternatives_NonExistentModel_ThrowsException()
        {
            // Arrange
            string modelName = "non_existent_model";
            int count = 2;

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                _engine.GetAlternatives(modelName, count));
        }

        [Fact]
        public void RecommendWithStrategy_AccuracyFirst_PrioritizesAccuracy()
        {
            // Arrange
            var inputShape = new Shape(new[] { 3, 224, 224 });
            var strategy = RecommendationStrategy.AccuracyFirst;

            // Act
            var recommendations = _engine.RecommendWithStrategy(
                inputShape,
                TaskType.ImageClassification,
                strategy,
                topN: 1);

            // Assert
            Assert.NotNull(recommendations);
            Assert.Single(recommendations);
            Assert.NotNull(recommendations[0].Model);
            Assert.Contains("accuracy", recommendations[0].Model.PerformanceMetrics.Keys);
        }

        [Fact]
        public void RecommendWithStrategy_PerformanceFirst_PrioritizesLatency()
        {
            // Arrange
            var inputShape = new Shape(new[] { 3, 224, 224 });
            var strategy = RecommendationStrategy.PerformanceFirst;

            // Act
            var recommendations = _engine.RecommendWithStrategy(
                inputShape,
                TaskType.ImageClassification,
                strategy,
                topN: 1);

            // Assert
            Assert.NotNull(recommendations);
            Assert.Single(recommendations);
            // PerformanceFirst should recommend smaller models
            Assert.True(recommendations[0].Model.NumParameters < 20_000_000,
                "PerformanceFirst should recommend smaller models");
        }

        [Fact]
        public void RecommendWithStrategy_MemoryConstrained_PrioritizesSmallModels()
        {
            // Arrange
            var inputShape = new Shape(new[] { 3, 224, 224 });
            var strategy = RecommendationStrategy.MemoryConstrained;

            // Act
            var recommendations = _engine.RecommendWithStrategy(
                inputShape,
                TaskType.ImageClassification,
                strategy,
                topN: 1);

            // Assert
            Assert.NotNull(recommendations);
            Assert.Single(recommendations);
            // MemoryConstrained should recommend the smallest model
            var allModels = _registry.ListByTask(TaskType.ImageClassification);
            var smallestModel = allModels.OrderBy(m => m.FileSizeBytes).First();
            Assert.Equal(smallestModel.Name, recommendations[0].Model.Name);
        }

        [Fact]
        public void LatencyEstimator_LargerModel_HigherLatency()
        {
            // Arrange
            var smallModel = _registry.Get("mobilenet_v2");
            var largeModel = _registry.Get("resnet50");

            // Act
            float smallLatency = _latencyEstimator.EstimateLatency(smallModel!, DeviceType.CPU);
            float largeLatency = _latencyEstimator.EstimateLatency(largeModel!, DeviceType.CPU);

            // Assert
            Assert.True(largeLatency > smallLatency,
                $"Large model latency {largeLatency} should be greater than small model latency {smallLatency}");
        }

        [Fact]
        public void LatencyEstimator_GPU_FasterThanCPU()
        {
            // Arrange
            var model = _registry.Get("resnet50");

            // Act
            float cpuLatency = _latencyEstimator.EstimateLatency(model!, DeviceType.CPU);
            float gpuLatency = _latencyEstimator.EstimateLatency(model!, DeviceType.GPU);

            // Assert
            Assert.True(cpuLatency > gpuLatency,
                $"CPU latency {cpuLatency} should be greater than GPU latency {gpuLatency}");
        }

        [Fact]
        public void LatencyEstimator_BatchSize_IncreasesLatency()
        {
            // Arrange
            var model = _registry.Get("resnet50");

            // Act
            float batch1Latency = _latencyEstimator.EstimateLatency(model!, DeviceType.CPU, batchSize: 1);
            float batch4Latency = _latencyEstimator.EstimateLatency(model!, DeviceType.CPU, batchSize: 4);

            // Assert
            Assert.True(batch4Latency > batch1Latency,
                $"Batch 4 latency {batch4Latency} should be greater than batch 1 latency {batch1Latency}");
        }

        [Fact]
        public void MemoryEstimator_LargerModel_HigherMemory()
        {
            // Arrange
            var smallModel = _registry.Get("mobilenet_v2");
            var largeModel = _registry.Get("bert_large");

            // Act
            float smallMemory = _memoryEstimator.EstimateMemory(smallModel!, DeviceType.CPU);
            float largeMemory = _memoryEstimator.EstimateMemory(largeModel!, DeviceType.CPU);

            // Assert
            Assert.True(largeMemory > smallMemory,
                $"Large model memory {largeMemory} should be greater than small model memory {smallMemory}");
        }

        [Fact]
        public void MemoryEstimator_BatchSize_IncreasesMemory()
        {
            // Arrange
            var model = _registry.Get("resnet50");

            // Act
            float batch1Memory = _memoryEstimator.EstimateMemory(model!, DeviceType.CPU, batchSize: 1);
            float batch4Memory = _memoryEstimator.EstimateMemory(model!, DeviceType.CPU, batchSize: 4);

            // Assert
            Assert.True(batch4Memory > batch1Memory,
                $"Batch 4 memory {batch4Memory} should be greater than batch 1 memory {batch1Memory}");
        }

        [Fact]
        public void ModelConstraints_Builder_CreatesValidConstraints()
        {
            // Arrange
            var inputShape = new Shape(new[] { 3, 224, 224 });

            // Act
            var constraints = ModelConstraints.Create()
                .WithInputShape(inputShape)
                .WithTask(TaskType.ImageClassification)
                .WithMaxLatency(50.0f)
                .WithMaxMemory(100 * 1024 * 1024)
                .WithMinAccuracy(0.70f)
                .WithMaxFileSize(200 * 1024 * 1024)
                .WithDevice(DeviceType.GPU)
                .WithBatchSize(4)
                .WithDeploymentEnvironment(DeploymentEnv.Cloud);

            // Assert
            Assert.NotNull(constraints);
            Assert.Equal(inputShape, constraints.InputShape);
            Assert.Equal(TaskType.ImageClassification, constraints.Task);
            Assert.Equal(50.0f, constraints.MaxLatency);
            Assert.Equal(100 * 1024 * 1024, constraints.MaxMemory);
            Assert.Equal(0.70f, constraints.MinAccuracy);
            Assert.Equal(200 * 1024 * 1024, constraints.MaxFileSize);
            Assert.Equal(DeviceType.GPU, constraints.Device);
            Assert.Equal(4, constraints.BatchSize);
            Assert.Equal(DeploymentEnv.Cloud, constraints.DeploymentEnvironment);
        }

        [Fact]
        public void ModelRecommendation_SatisfiesConstraints_ReturnsTrueWhenValid()
        {
            // Arrange
            var inputShape = new Shape(new[] { 3, 224, 224 });
            var constraints = ModelConstraints.Create()
                .WithInputShape(inputShape)
                .WithTask(TaskType.ImageClassification)
                .WithMaxLatency(100.0f)
                .WithMaxMemory(200 * 1024 * 1024);

            var recommendation = _engine.RecommendFor(inputShape, TaskType.ImageClassification, constraints);

            // Act
            bool satisfies = recommendation.SatisfiesConstraints(constraints);

            // Assert
            Assert.True(satisfies);
        }

        [Fact]
        public void ModelRecommendation_SatisfiesConstraints_ReturnsFalseWhenViolated()
        {
            // Arrange
            var inputShape = new Shape(new[] { 3, 224, 224 });
            var constraints = ModelConstraints.Create()
                .WithInputShape(inputShape)
                .WithTask(TaskType.ImageClassification)
                .WithMaxLatency(0.001f); // Unrealistic constraint

            var recommendation = _engine.RecommendFor(inputShape, TaskType.ImageClassification);

            // Act
            bool satisfies = recommendation.SatisfiesConstraints(constraints);

            // Assert
            Assert.False(satisfies);
        }

        [Fact]
        public void Shape_MatchesExactly_ReturnsTrueForMatchingShapes()
        {
            // Arrange
            var shape1 = new Shape(new[] { 3, 224, 224 });
            var shape2 = new Shape(new[] { 3, 224, 224 });

            // Act
            bool matches = shape1.MatchesExactly(shape2);

            // Assert
            Assert.True(matches);
        }

        [Fact]
        public void Shape_MatchesExactly_ReturnsFalseForDifferentShapes()
        {
            // Arrange
            var shape1 = new Shape(new[] { 3, 224, 224 });
            var shape2 = new Shape(new[] { 3, 256, 256 });

            // Act
            bool matches = shape1.MatchesExactly(shape2);

            // Assert
            Assert.False(matches);
        }

        [Fact]
        public void Shape_IsCompatibleWith_AllowsVariableDimensions()
        {
            // Arrange
            var shape1 = new Shape(new[] { -1, 224, 224 }); // Variable batch size
            var shape2 = new Shape(new[] { 4, 224, 224 });

            // Act
            bool compatible = shape1.IsCompatibleWith(shape2);

            // Assert
            Assert.True(compatible);
        }

        [Fact]
        public void RecommendFor_InvalidTask_ThrowsException()
        {
            // Arrange
            var inputShape = new Shape(new[] { 3, 224, 224 });
            var invalidTask = TaskType.Regression; // No regression models in registry

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                _engine.RecommendFor(inputShape, invalidTask));
        }

        public void Dispose()
        {
            _registry.Clear();
        }
    }
}
