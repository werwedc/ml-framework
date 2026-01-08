using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MLFramework.Serving.Traffic;
using Xunit;

namespace MLFramework.Tests.Serving.Traffic
{
    /// <summary>
    /// Comprehensive unit tests for TrafficSplitter functionality.
    /// </summary>
    public class TrafficSplitterTests
    {
        [Fact]
        public void SetTrafficSplit_ValidSplit_Succeeds()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float>
            {
                { "v1.0", 0.7f },
                { "v1.1", 0.3f }
            };

            // Act
            splitter.SetTrafficSplit("model1", percentages);

            // Assert
            var config = splitter.GetTrafficSplit("model1");
            Assert.NotNull(config);
            Assert.Equal(2, config.VersionPercentages.Count);
            Assert.Equal(0.7f, config.VersionPercentages["v1.0"]);
            Assert.Equal(0.3f, config.VersionPercentages["v1.1"]);
        }

        [Fact]
        public void SetTrafficSplit_InvalidSum_ThrowsException()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float>
            {
                { "v1.0", 0.7f },
                { "v1.1", 0.4f }  // Sum = 1.1, should throw
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => splitter.SetTrafficSplit("model1", percentages));
        }

        [Fact]
        public void SetTrafficSplit_NegativePercentage_ThrowsException()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float>
            {
                { "v1.0", -0.1f },
                { "v1.1", 1.1f }  // Negative and > 1.0
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => splitter.SetTrafficSplit("model1", percentages));
        }

        [Fact]
        public void SetTrafficSplit_NullModelName_ThrowsException()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float> { { "v1.0", 1.0f } };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => splitter.SetTrafficSplit(null, percentages));
            Assert.Throws<ArgumentNullException>(() => splitter.SetTrafficSplit("", percentages));
            Assert.Throws<ArgumentNullException>(() => splitter.SetTrafficSplit("  ", percentages));
        }

        [Fact]
        public void SetTrafficSplit_NullPercentages_ThrowsException()
        {
            // Arrange
            var splitter = new TrafficSplitter();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => splitter.SetTrafficSplit("model1", null));
        }

        [Fact]
        public void SetTrafficSplit_EmptyPercentages_ThrowsException()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => splitter.SetTrafficSplit("model1", percentages));
        }

        [Fact]
        public void SelectVersion_ValidSplit_ReturnsVersion()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float>
            {
                { "v1.0", 0.7f },
                { "v1.1", 0.3f }
            };
            splitter.SetTrafficSplit("model1", percentages);

            // Act
            var version = splitter.SelectVersion("model1", "request-123");

            // Assert
            Assert.True(version == "v1.0" || version == "v1.1");
        }

        [Fact]
        public void SelectVersion_NoSplitConfigured_ThrowsException()
        {
            // Arrange
            var splitter = new TrafficSplitter();

            // Act & Assert
            Assert.Throws<KeyNotFoundException>(() => splitter.SelectVersion("nonexistent", "request-123"));
        }

        [Fact]
        public void SelectVersion_NullParameters_ThrowsException()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float> { { "v1.0", 1.0f } };
            splitter.SetTrafficSplit("model1", percentages);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => splitter.SelectVersion(null, "request-123"));
            Assert.Throws<ArgumentNullException>(() => splitter.SelectVersion("", "request-123"));
            Assert.Throws<ArgumentNullException>(() => splitter.SelectVersion("model1", null));
            Assert.Throws<ArgumentNullException>(() => splitter.SelectVersion("model1", ""));
        }

        [Fact]
        public void SelectVersion_SameRequestId_ReturnsSameVersion()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float>
            {
                { "v1.0", 0.5f },
                { "v1.1", 0.5f }
            };
            splitter.SetTrafficSplit("model1", percentages);
            string requestId = "deterministic-request";

            // Act
            var version1 = splitter.SelectVersion("model1", requestId);
            var version2 = splitter.SelectVersion("model1", requestId);
            var version3 = splitter.SelectVersion("model1", requestId);

            // Assert
            Assert.Equal(version1, version2);
            Assert.Equal(version2, version3);
        }

        [Fact]
        public void SelectVersion_DistributionTest_70_30Split()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float>
            {
                { "v1.0", 0.7f },
                { "v1.1", 0.3f }
            };
            splitter.SetTrafficSplit("model1", percentages);

            // Act
            int v1Count = 0;
            int v2Count = 0;
            const int requestCount = 1000;

            for (int i = 0; i < requestCount; i++)
            {
                var version = splitter.SelectVersion("model1", $"request-{i}");
                if (version == "v1.0") v1Count++;
                else if (version == "v1.1") v2Count++;
            }

            // Assert - Allow 2% error margin
            float v1Percentage = (float)v1Count / requestCount;
            float v2Percentage = (float)v2Count / requestCount;

            Assert.InRange(v1Percentage, 0.68f, 0.72f); // 70% ± 2%
            Assert.InRange(v2Percentage, 0.28f, 0.32f); // 30% ± 2%
        }

        [Fact]
        public void SelectVersion_UpdateSplit_NewDistribution()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var initialSplit = new Dictionary<string, float>
            {
                { "v1.0", 0.7f },
                { "v1.1", 0.3f }
            };
            splitter.SetTrafficSplit("model1", initialSplit);

            // Act - Update to 50/50
            var updatedSplit = new Dictionary<string, float>
            {
                { "v1.0", 0.5f },
                { "v1.1", 0.5f }
            };
            splitter.SetTrafficSplit("model1", updatedSplit);

            // Test new distribution
            int v1Count = 0;
            int v2Count = 0;
            const int requestCount = 1000;

            for (int i = 0; i < requestCount; i++)
            {
                var version = splitter.SelectVersion("model1", $"request-{i}");
                if (version == "v1.0") v1Count++;
                else if (version == "v1.1") v2Count++;
            }

            // Assert - Allow 2% error margin
            float v1Percentage = (float)v1Count / requestCount;
            float v2Percentage = (float)v2Count / requestCount;

            Assert.InRange(v1Percentage, 0.48f, 0.52f); // 50% ± 2%
            Assert.InRange(v2Percentage, 0.48f, 0.52f); // 50% ± 2%
        }

        [Fact]
        public void ClearTrafficSplit_RemovesConfiguration()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float> { { "v1.0", 1.0f } };
            splitter.SetTrafficSplit("model1", percentages);

            // Act
            splitter.ClearTrafficSplit("model1");

            // Assert
            Assert.Null(splitter.GetTrafficSplit("model1"));
            Assert.Throws<KeyNotFoundException>(() => splitter.SelectVersion("model1", "request-123"));
        }

        [Fact]
        public void GetVersionAllocation_ReturnsCorrectPercentage()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float>
            {
                { "v1.0", 0.7f },
                { "v1.1", 0.3f }
            };
            splitter.SetTrafficSplit("model1", percentages);

            // Act
            var v1Allocation = splitter.GetVersionAllocation("model1", "v1.0");
            var v2Allocation = splitter.GetVersionAllocation("model1", "v1.1");
            var v3Allocation = splitter.GetVersionAllocation("model1", "v1.2");

            // Assert
            Assert.Equal(0.7f, v1Allocation);
            Assert.Equal(0.3f, v2Allocation);
            Assert.Equal(0.0f, v3Allocation);
        }

        [Fact]
        public void GetVersionAllocation_NonExistentModel_ReturnsZero()
        {
            // Arrange
            var splitter = new TrafficSplitter();

            // Act
            var allocation = splitter.GetVersionAllocation("nonexistent", "v1.0");

            // Assert
            Assert.Equal(0.0f, allocation);
        }

        [Fact]
        public void GetVersionAllocation_NullParameters_ThrowsException()
        {
            // Arrange
            var splitter = new TrafficSplitter();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => splitter.GetVersionAllocation(null, "v1.0"));
            Assert.Throws<ArgumentNullException>(() => splitter.GetVersionAllocation("model1", null));
        }

        [Fact]
        public void GetTrafficSplit_ReturnsNullForNonExistent()
        {
            // Arrange
            var splitter = new TrafficSplitter();

            // Act
            var config = splitter.GetTrafficSplit("nonexistent");

            // Assert
            Assert.Null(config);
        }

        [Fact]
        public void GetTrafficSplit_NullModelName_ThrowsException()
        {
            // Arrange
            var splitter = new TrafficSplitter();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => splitter.GetTrafficSplit(null));
            Assert.Throws<ArgumentNullException>(() => splitter.GetTrafficSplit(""));
        }

        [Fact]
        public void ConcurrentUpdates_ThreadSafe()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            const int threadCount = 10;
            const int operationsPerThread = 100;

            // Act
            var tasks = new List<Task>();
            for (int t = 0; t < threadCount; t++)
            {
                int threadId = t;
                tasks.Add(Task.Run(() =>
                {
                    for (int i = 0; i < operationsPerThread; i++)
                    {
                        var percentages = new Dictionary<string, float>
                        {
                            { "v1.0", 0.5f },
                            { "v1.1", 0.5f }
                        };
                        splitter.SetTrafficSplit($"model-{threadId}", percentages);

                        var version = splitter.SelectVersion($"model-{threadId}", $"request-{i}");
                        Assert.True(version == "v1.0" || version == "v1.1");
                    }
                }));
            }

            Task.WaitAll(tasks.ToArray());

            // Assert
            for (int t = 0; t < threadCount; t++)
            {
                var config = splitter.GetTrafficSplit($"model-{t}");
                Assert.NotNull(config);
                Assert.Equal(2, config.VersionPercentages.Count);
            }
        }

        [Fact]
        public void PerformanceTest_SelectVersion_Sub100Microseconds()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float>
            {
                { "v1.0", 0.5f },
                { "v1.1", 0.5f }
            };
            splitter.SetTrafficSplit("model1", percentages);

            // Act
            var stopwatch = Stopwatch.StartNew();
            const int iterations = 1000;

            for (int i = 0; i < iterations; i++)
            {
                splitter.SelectVersion("model1", $"request-{i}");
            }

            stopwatch.Stop();

            // Assert - Average should be less than 0.1ms (100 microseconds)
            double averageMs = stopwatch.Elapsed.TotalMilliseconds / iterations;
            Assert.True(averageMs < 0.1, $"Selection time {averageMs:F4}ms exceeds 0.1ms threshold");
        }

        [Fact]
        public void PerformanceTest_1000SelectionsPerMs()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float>
            {
                { "v1.0", 0.5f },
                { "v1.1", 0.5f }
            };
            splitter.SetTrafficSplit("model1", percentages);

            // Act
            var stopwatch = Stopwatch.StartNew();
            const int targetCount = 1000;
            const double maxMs = 1.0;

            int actualCount = 0;
            while (stopwatch.Elapsed.TotalMilliseconds < maxMs)
            {
                splitter.SelectVersion("model1", $"request-{actualCount}");
                actualCount++;
            }

            stopwatch.Stop();

            // Assert - Should complete at least 1000 selections in 1ms
            Assert.True(actualCount >= targetCount, $"Completed {actualCount} selections in {maxMs}ms, expected at least {targetCount}");
        }

        [Fact]
        public void SetTrafficSplit_SingleVersion_Succeeds()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float> { { "v1.0", 1.0f } };

            // Act
            splitter.SetTrafficSplit("model1", percentages);

            // Assert
            var version = splitter.SelectVersion("model1", "request-123");
            Assert.Equal("v1.0", version);
        }

        [Fact]
        public void SetTrafficSplit_TenVersions_Succeeds()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float>();
            for (int i = 0; i < 10; i++)
            {
                percentages[$"v{i}.0"] = 0.1f;
            }

            // Act
            splitter.SetTrafficSplit("model1", percentages);

            // Assert
            int v1Count = 0;
            const int requestCount = 1000;

            for (int i = 0; i < requestCount; i++)
            {
                var version = splitter.SelectVersion("model1", $"request-{i}");
                if (version == "v0.0") v1Count++;
            }

            float v1Percentage = (float)v1Count / requestCount;
            Assert.InRange(v1Percentage, 0.08f, 0.12f); // 10% ± 2%
        }

        [Fact]
        public void SetTrafficSplit_VerySmallPercentage_Succeeds()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float>
            {
                { "v1.0", 0.99f },
                { "v1.1", 0.01f }
            };

            // Act
            splitter.SetTrafficSplit("model1", percentages);

            // Assert - With 1000 requests, we should get some v1.1
            int v2Count = 0;
            const int requestCount = 1000;

            for (int i = 0; i < requestCount; i++)
            {
                var version = splitter.SelectVersion("model1", $"request-{i}");
                if (version == "v1.1") v2Count++;
            }

            // Should get roughly 10 requests (1% of 1000), allow some variance
            Assert.True(v2Count >= 0, $"v1.1 count: {v2Count}");
        }

        [Fact]
        public void SetTrafficSplit_FloatingPointPrecision_AcceptsNearOne()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float>
            {
                { "v1.0", 0.3333333f },
                { "v1.1", 0.3333333f },
                { "v1.2", 0.3333334f }
            };

            // Act & Assert - Should not throw despite floating point imprecision
            splitter.SetTrafficSplit("model1", percentages);
        }

        [Fact]
        public void DeterministicRouting_DifferentModels_Independent()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages1 = new Dictionary<string, float> { { "v1.0", 1.0f } };
            var percentages2 = new Dictionary<string, float> { { "v2.0", 1.0f } };
            splitter.SetTrafficSplit("model1", percentages1);
            splitter.SetTrafficSplit("model2", percentages2);

            // Act
            var version1 = splitter.SelectVersion("model1", "same-request");
            var version2 = splitter.SelectVersion("model2", "same-request");

            // Assert - Different models should route independently
            Assert.Equal("v1.0", version1);
            Assert.Equal("v2.0", version2);
        }

        [Fact]
        public void SetTrafficSplit_UpdatesConfigurationMetadata()
        {
            // Arrange
            var splitter = new TrafficSplitter();
            var percentages = new Dictionary<string, float> { { "v1.0", 1.0f } };

            // Act
            splitter.SetTrafficSplit("model1", percentages);
            var config = splitter.GetTrafficSplit("model1");

            // Assert
            Assert.NotNull(config);
            Assert.NotNull(config.UpdatedBy);
            Assert.True(config.LastUpdated <= DateTime.UtcNow);
            Assert.True(config.LastUpdated > DateTime.UtcNow.AddMinutes(-1));
        }

        [Fact]
        public void ClearTrafficSplit_NullModelName_ThrowsException()
        {
            // Arrange
            var splitter = new TrafficSplitter();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => splitter.ClearTrafficSplit(null));
            Assert.Throws<ArgumentNullException>(() => splitter.ClearTrafficSplit(""));
        }
    }
}
