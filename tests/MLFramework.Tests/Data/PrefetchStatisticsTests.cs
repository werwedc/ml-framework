using Xunit;
using MLFramework.Data;

namespace MLFramework.Tests.Data
{
    public class PrefetchStatisticsTests
    {
        [Fact]
        public void Constructor_DefaultValues_AllZero()
        {
            // Arrange & Act
            var stats = new PrefetchStatistics();

            // Assert
            Assert.Equal(0, stats.CacheHits);
            Assert.Equal(0, stats.CacheMisses);
            Assert.Equal(0.0, stats.CacheHitRate);
            Assert.Equal(0.0, stats.AverageLatencyMs);
            Assert.Equal(0, stats.RefillCount);
            Assert.Equal(0, stats.StarvationCount);
            Assert.Equal(0, stats.TotalRequests);
        }

        [Fact]
        public void CacheHitRate_OnlyHits_Returns100Percent()
        {
            // Arrange
            var stats = new PrefetchStatistics
            {
                CacheHits = 10,
                CacheMisses = 0
            };

            // Act & Assert
            Assert.Equal(1.0, stats.CacheHitRate);
        }

        [Fact]
        public void CacheHitRate_OnlyMisses_Returns0Percent()
        {
            // Arrange
            var stats = new PrefetchStatistics
            {
                CacheHits = 0,
                CacheMisses = 10
            };

            // Act & Assert
            Assert.Equal(0.0, stats.CacheHitRate);
        }

        [Fact]
        public void CacheHitRate_MixedHitsAndMisses_CalculatesCorrectly()
        {
            // Arrange
            var stats = new PrefetchStatistics
            {
                CacheHits = 75,
                CacheMisses = 25
            };

            // Act & Assert
            Assert.Equal(0.75, stats.CacheHitRate);
        }

        [Fact]
        public void TotalRequests_SumOfHitsAndMisses_CalculatesCorrectly()
        {
            // Arrange
            var stats = new PrefetchStatistics
            {
                CacheHits = 10,
                CacheMisses = 5
            };

            // Act & Assert
            Assert.Equal(15, stats.TotalRequests);
        }

        [Fact]
        public void Reset_AllNonZeroValues_ResetsToZero()
        {
            // Arrange
            var stats = new PrefetchStatistics
            {
                CacheHits = 100,
                CacheMisses = 20,
                AverageLatencyMs = 50.5,
                RefillCount = 5,
                StarvationCount = 2
            };

            // Act
            stats.Reset();

            // Assert
            Assert.Equal(0, stats.CacheHits);
            Assert.Equal(0, stats.CacheMisses);
            Assert.Equal(0.0, stats.CacheHitRate);
            Assert.Equal(0.0, stats.AverageLatencyMs);
            Assert.Equal(0, stats.RefillCount);
            Assert.Equal(0, stats.StarvationCount);
            Assert.Equal(0, stats.TotalRequests);
        }

        [Fact]
        public void Reset_AlreadyZeroValues_StaysZero()
        {
            // Arrange
            var stats = new PrefetchStatistics();

            // Act
            stats.Reset();

            // Assert
            Assert.Equal(0, stats.CacheHits);
            Assert.Equal(0, stats.CacheMisses);
            Assert.Equal(0.0, stats.CacheHitRate);
            Assert.Equal(0.0, stats.AverageLatencyMs);
            Assert.Equal(0, stats.RefillCount);
            Assert.Equal(0, stats.StarvationCount);
        }

        [Fact]
        public void ToString_ContainsAllStatistics()
        {
            // Arrange
            var stats = new PrefetchStatistics
            {
                CacheHits = 75,
                CacheMisses = 25,
                AverageLatencyMs = 10.5,
                RefillCount = 3,
                StarvationCount = 1
            };

            // Act
            var str = stats.ToString();

            // Assert
            Assert.Contains("CacheHits=75", str);
            Assert.Contains("CacheMisses=25", str);
            Assert.Contains("CacheHitRate=75.00%", str);
            Assert.Contains("AvgLatency=10.50ms", str);
            Assert.Contains("RefillCount=3", str);
            Assert.Contains("StarvationCount=1", str);
        }

        [Fact]
        public void CacheHitRate_NoRequests_ReturnsZero()
        {
            // Arrange
            var stats = new PrefetchStatistics
            {
                CacheHits = 0,
                CacheMisses = 0
            };

            // Act & Assert
            Assert.Equal(0.0, stats.CacheHitRate);
        }

        [Fact]
        public void AllProperties_SetAndGet_WorkCorrectly()
        {
            // Arrange
            var stats = new PrefetchStatistics();

            // Act
            stats.CacheHits = 50;
            stats.CacheMisses = 10;
            stats.AverageLatencyMs = 25.75;
            stats.RefillCount = 7;
            stats.StarvationCount = 3;

            // Assert
            Assert.Equal(50, stats.CacheHits);
            Assert.Equal(10, stats.CacheMisses);
            Assert.Equal(25.75, stats.AverageLatencyMs);
            Assert.Equal(7, stats.RefillCount);
            Assert.Equal(3, stats.StarvationCount);
            Assert.Equal(60, stats.TotalRequests);
            Assert.Equal(0.8333333333333334, stats.CacheHitRate, precision: 10);
        }
    }
}
