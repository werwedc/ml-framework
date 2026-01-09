using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.ModelZoo;
using MLFramework.ModelZoo.Discovery;
using Xunit;

namespace ModelZooTests.Discovery
{
    /// <summary>
    /// Unit tests for ModelSearchService.
    /// </summary>
    public class SearchTests : IDisposable
    {
        private readonly ModelRegistry _registry;
        private readonly ModelSearchService _searchService;

        public SearchTests()
        {
            _registry = new ModelRegistry();
            _searchService = new ModelSearchService(_registry);
            SetupTestModels();
        }

        private void SetupTestModels()
        {
            // ResNet-50 (Image Classification)
            var resnet50 = new ModelMetadata
            {
                Name = "resnet50",
                Version = "1.0.0",
                Architecture = "ResNet",
                Task = TaskType.ImageClassification,
                PretrainedOn = "ImageNet",
                PerformanceMetrics = new Dictionary<string, double>
                {
                    { "top1", 0.76 },
                    { "top5", 0.93 }
                },
                InputShape = new[] { 3, 224, 224 },
                OutputShape = new[] { 1000 },
                NumParameters = 25_600_000,
                FileSizeBytes = 98 * 1024 * 1024, // 98 MB
                License = "MIT",
                Sha256Checksum = "abc123",
                DownloadUrl = "https://example.com/resnet50.pt"
            };

            // ResNet-18 (smaller, faster)
            var resnet18 = new ModelMetadata
            {
                Name = "resnet18",
                Version = "1.0.0",
                Architecture = "ResNet",
                Task = TaskType.ImageClassification,
                PretrainedOn = "ImageNet",
                PerformanceMetrics = new Dictionary<string, double>
                {
                    { "top1", 0.70 },
                    { "top5", 0.89 }
                },
                InputShape = new[] { 3, 224, 224 },
                OutputShape = new[] { 1000 },
                NumParameters = 11_700_000,
                FileSizeBytes = 45 * 1024 * 1024, // 45 MB
                License = "MIT",
                Sha256Checksum = "def456",
                DownloadUrl = "https://example.com/resnet18.pt"
            };

            // BERT (Text Classification)
            var bert = new ModelMetadata
            {
                Name = "bert-base-uncased",
                Version = "1.0.0",
                Architecture = "BERT",
                Task = TaskType.TextClassification,
                PretrainedOn = "BookCorpus+Wiki",
                PerformanceMetrics = new Dictionary<string, double>
                {
                    { "accuracy", 0.85 }
                },
                InputShape = new[] { -1, 512 }, // Sequence length 512
                OutputShape = new[] { -1, 2 }, // Binary classification
                NumParameters = 110_000_000,
                FileSizeBytes = 420 * 1024 * 1024, // 420 MB
                License = "Apache-2.0",
                Sha256Checksum = "ghi789",
                DownloadUrl = "https://example.com/bert.pt"
            };

            // YOLOv5 (Object Detection)
            var yolo = new ModelMetadata
            {
                Name = "yolov5s",
                Version = "6.0.0",
                Architecture = "YOLO",
                Task = TaskType.ObjectDetection,
                PretrainedOn = "COCO",
                PerformanceMetrics = new Dictionary<string, double>
                {
                    { "map50", 0.56 }
                },
                InputShape = new[] { 3, 640, 640 },
                OutputShape = new[] { 25200, 85 },
                NumParameters = 7_200_000,
                FileSizeBytes = 14 * 1024 * 1024, // 14 MB
                License = "GPL-3.0",
                Sha256Checksum = "jkl012",
                DownloadUrl = "https://example.com/yolov5s.pt"
            };

            // EfficientNet (High accuracy)
            var efficientnet = new ModelMetadata
            {
                Name = "efficientnet-b4",
                Version = "1.0.0",
                Architecture = "EfficientNet",
                Task = TaskType.ImageClassification,
                PretrainedOn = "ImageNet",
                PerformanceMetrics = new Dictionary<string, double>
                {
                    { "top1", 0.82 },
                    { "top5", 0.96 }
                },
                InputShape = new[] { 3, 380, 380 },
                OutputShape = new[] { 1000 },
                NumParameters = 19_300_000,
                FileSizeBytes = 75 * 1024 * 1024, // 75 MB
                License = "Apache-2.0",
                Sha256Checksum = "mno345",
                DownloadUrl = "https://example.com/efficientnet-b4.pt"
            };

            _registry.RegisterModel(resnet50);
            _registry.RegisterModel(resnet18);
            _registry.RegisterModel(bert);
            _registry.RegisterModel(yolo);
            _registry.RegisterModel(efficientnet);
        }

        [Fact]
        public void Search_ByTask_ReturnsMatchingModels()
        {
            // Arrange
            var query = new ModelSearchQuery
            {
                Task = TaskType.ImageClassification
            };

            // Act
            var results = _searchService.Search(query);

            // Assert
            Assert.Equal(3, results.Count);
            Assert.All(results, r => Assert.Equal(TaskType.ImageClassification, r.Model.Task));
        }

        [Fact]
        public void Search_ByAccuracyRange_ReturnsMatchingModels()
        {
            // Arrange
            var query = new ModelSearchQuery
            {
                MinAccuracy = 0.75
            };

            // Act
            var results = _searchService.Search(query);

            // Assert
            Assert.All(results, r =>
            {
                var accuracy = GetPrimaryAccuracy(r.Model);
                Assert.True(accuracy >= 0.75);
            });
        }

        [Fact]
        public void Search_ByArchitecture_ReturnsMatchingModels()
        {
            // Arrange
            var query = new ModelSearchQuery
            {
                Architecture = "ResNet"
            };

            // Act
            var results = _searchService.Search(query);

            // Assert
            Assert.Equal(2, results.Count);
            Assert.All(results, r => Assert.Contains("ResNet", r.Model.Architecture));
        }

        [Fact]
        public void Search_WithMultipleFilters_ReturnsMatchingModels()
        {
            // Arrange
            var query = new ModelSearchQuery
            {
                Task = TaskType.ImageClassification,
                Architecture = "ResNet",
                MinAccuracy = 0.72
            };

            // Act
            var results = _searchService.Search(query);

            // Assert
            Assert.Single(results);
            Assert.Equal("resnet50", results[0].Model.Name);
        }

        [Fact]
        public void Search_WithCustomFilters_ReturnsMatchingModels()
        {
            // Arrange
            var query = new ModelSearchQuery
            {
                Task = TaskType.ImageClassification,
                CustomFilters = new Dictionary<string, Func<ModelMetadata, bool>>
                {
                    { "SmallSize", m => m.FileSizeBytes < 50 * 1024 * 1024 }
                }
            };

            // Act
            var results = _searchService.Search(query);

            // Assert
            Assert.Single(results);
            Assert.Equal("resnet18", results[0].Model.Name);
        }

        [Fact]
        public void Search_SortByAccuracy_Descending_ReturnsCorrectOrder()
        {
            // Arrange
            var query = new ModelSearchQuery
            {
                Task = TaskType.ImageClassification,
                SortBy = SearchSortBy.Accuracy,
                SortDescending = true
            };

            // Act
            var results = _searchService.Search(query);

            // Assert
            Assert.Equal("efficientnet-b4", results[0].Model.Name);
            Assert.Equal("resnet50", results[1].Model.Name);
            Assert.Equal("resnet18", results[2].Model.Name);
        }

        [Fact]
        public void Search_SortBySize_Ascending_ReturnsCorrectOrder()
        {
            // Arrange
            var query = new ModelSearchQuery
            {
                Task = TaskType.ImageClassification,
                SortBy = SearchSortBy.Size,
                SortDescending = false
            };

            // Act
            var results = _searchService.Search(query);

            // Assert
            Assert.Equal("resnet18", results[0].Model.Name);
            Assert.Equal("efficientnet-b4", results[1].Model.Name);
            Assert.Equal("resnet50", results[2].Model.Name);
        }

        [Fact]
        public void Search_WithLimit_ReturnsLimitedResults()
        {
            // Arrange
            var query = new ModelSearchQuery
            {
                Task = TaskType.ImageClassification,
                Limit = 2
            };

            // Act
            var results = _searchService.Search(query);

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Search_NoResults_ReturnsEmptyList()
        {
            // Arrange
            var query = new ModelSearchQuery
            {
                Task = TaskType.Regression
            };

            // Act
            var results = _searchService.Search(query);

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Search_MatchScore_IsCalculatedCorrectly()
        {
            // Arrange
            var query = new ModelSearchQuery
            {
                MinAccuracy = 0.70
            };

            // Act
            var results = _searchService.Search(query);

            // Assert
            Assert.All(results, r =>
            {
                Assert.InRange(r.MatchScore, 0.0, 1.0);
                Assert.NotEmpty(r.MatchReasons);
            });
        }

        [Fact]
        public void SearchByTask_QuickSearch_ReturnsMatchingModels()
        {
            // Act
            var results = _searchService.SearchByTask(TaskType.ImageClassification);

            // Assert
            Assert.Equal(3, results.Count);
            Assert.All(results, r => Assert.Equal(TaskType.ImageClassification, r.Model.Task));
        }

        [Fact]
        public void SearchByAccuracy_WithMinAccuracy_ReturnsMatchingModels()
        {
            // Act
            var results = _searchService.SearchByAccuracy(0.80);

            // Assert
            Assert.Equal(2, results.Count);
            Assert.All(results, r =>
            {
                var accuracy = GetPrimaryAccuracy(r.Model);
                Assert.True(accuracy >= 0.80);
            });
        }

        [Fact]
        public void SearchByArchitecture_ReturnsMatchingModels()
        {
            // Act
            var results = _searchService.SearchByArchitecture("ResNet");

            // Assert
            Assert.Equal(2, results.Count);
            Assert.All(results, r => Assert.Contains("ResNet", r.Model.Architecture));
        }

        [Fact]
        public void SearchBySize_WithMaxSize_ReturnsMatchingModels()
        {
            // Act
            var results = _searchService.SearchBySize(50 * 1024 * 1024, TaskType.ImageClassification);

            // Assert
            Assert.Single(results);
            Assert.Equal("resnet18", results[0].Model.Name);
        }

        [Fact]
        public void AdvancedSearch_WithQueryBuilder_ReturnsCorrectResults()
        {
            // Act
            var results = _searchService.AdvancedSearch(builder =>
                builder.WithTask(TaskType.ImageClassification)
                       .WithAccuracyRange(0.70, 0.85)
                       .WithLicense("MIT")
                       .SortBy(SearchSortBy.Accuracy)
                       .WithLimit(10));

            // Assert
            Assert.Equal(2, results.Count);
            Assert.All(results, r => Assert.Equal("MIT", r.Model.License));
        }

        [Fact]
        public void ModelSearchQuery_Validation_ThrowsOnInvalidRange()
        {
            // Arrange
            var query = new ModelSearchQuery
            {
                MinAccuracy = 0.9,
                MaxAccuracy = 0.8
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => query.Validate());
        }

        [Fact]
        public void ModelSearchQuery_Validation_ThrowsOnInvalidLimit()
        {
            // Arrange
            var query = new ModelSearchQuery
            {
                Limit = 0
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => query.Validate());
        }

        [Fact]
        public void ModelSearchQueryBuilder_BuildsCorrectQuery()
        {
            // Act
            var query = new ModelSearchQueryBuilder()
                .WithTask(TaskType.ImageClassification)
                .WithArchitecture("ResNet")
                .WithMinAccuracy(0.75)
                .SortBy(SearchSortBy.Accuracy)
                .WithLimit(5)
                .Build();

            // Assert
            Assert.Equal(TaskType.ImageClassification, query.Task);
            Assert.Equal("ResNet", query.Architecture);
            Assert.Equal(0.75, query.MinAccuracy);
            Assert.Equal(SearchSortBy.Accuracy, query.SortBy);
            Assert.Equal(5, query.Limit);
        }

        [Fact]
        public void Search_WithInputShapeFilter_ReturnsMatchingModels()
        {
            // Arrange
            var query = new ModelSearchQuery
            {
                InputShape = new[] { 3, 224, 224 }
            };

            // Act
            var results = _searchService.Search(query);

            // Assert
            Assert.Equal(2, results.Count);
            Assert.All(results, r =>
            {
                Assert.Equal(3, r.Model.InputShape.Length);
                Assert.Equal(224, r.Model.InputShape[1]);
                Assert.Equal(224, r.Model.InputShape[2]);
            });
        }

        [Fact]
        public void Search_WithLicenseFilter_ReturnsMatchingModels()
        {
            // Arrange
            var query = new ModelSearchQuery
            {
                Task = TaskType.ImageClassification,
                License = "MIT"
            };

            // Act
            var results = _searchService.Search(query);

            // Assert
            Assert.Equal(2, results.Count);
            Assert.All(results, r => Assert.Equal("MIT", r.Model.License));
        }

        private double GetPrimaryAccuracy(ModelMetadata model)
        {
            if (model.PerformanceMetrics.TryGetValue("accuracy", out var accuracy))
            {
                return accuracy;
            }
            if (model.PerformanceMetrics.TryGetValue("top1", out var top1))
            {
                return top1;
            }
            if (model.PerformanceMetrics.TryGetValue("top5", out var top5))
            {
                return top5;
            }
            return 0.0;
        }

        public void Dispose()
        {
            // Cleanup if needed
        }
    }
}
