using Xunit;
using MLFramework.Data;

namespace MLFramework.Tests.Data
{
    /// <summary>
    /// Unit tests for PoolStatistics.
    /// </summary>
    public class PoolStatisticsTests
    {
        [Fact]
        public void NewStatistics_HasZeroValues()
        {
            // Arrange & Act
            var stats = new PoolStatistics();

            // Assert
            Assert.Equal(0, stats.RentCount);
            Assert.Equal(0, stats.ReturnCount);
            Assert.Equal(0, stats.MissCount);
            Assert.Equal(0, stats.DiscardCount);
            Assert.Equal(0.0, stats.HitRate);
        }

        [Fact]
        public void HitRate_WithNoRents_ReturnsZero()
        {
            // Arrange
            var stats = new PoolStatistics();

            // Act
            var hitRate = stats.HitRate;

            // Assert
            Assert.Equal(0.0, hitRate);
        }

        [Fact]
        public void HitRate_WithNoMisses_ReturnsOne()
        {
            // Arrange
            var stats = new PoolStatistics();

            // Act
            for (int i = 0; i < 10; i++)
            {
                stats.IncrementRent();
            }

            // Assert
            Assert.Equal(1.0, stats.HitRate);
        }

        [Fact]
        public void HitRate_WithSomeMisses_ReturnsCorrectValue()
        {
            // Arrange
            var stats = new PoolStatistics();

            // Act
            // 10 rents, 2 misses = 8 hits = 80% hit rate
            for (int i = 0; i < 10; i++)
            {
                stats.IncrementRent();
            }
            for (int i = 0; i < 2; i++)
            {
                stats.IncrementMiss();
            }

            // Assert
            Assert.Equal(0.8, stats.HitRate, 4); // 4 decimal precision
        }

        [Fact]
        public void IncrementRent_IncrementsRentCount()
        {
            // Arrange
            var stats = new PoolStatistics();

            // Act
            stats.IncrementRent();
            stats.IncrementRent();
            stats.IncrementRent();

            // Assert
            Assert.Equal(3, stats.RentCount);
        }

        [Fact]
        public void IncrementReturn_IncrementsReturnCount()
        {
            // Arrange
            var stats = new PoolStatistics();

            // Act
            stats.IncrementReturn();
            stats.IncrementReturn();
            stats.IncrementReturn();

            // Assert
            Assert.Equal(3, stats.ReturnCount);
        }

        [Fact]
        public void IncrementMiss_IncrementsMissCount()
        {
            // Arrange
            var stats = new PoolStatistics();

            // Act
            stats.IncrementMiss();
            stats.IncrementMiss();
            stats.IncrementMiss();

            // Assert
            Assert.Equal(3, stats.MissCount);
        }

        [Fact]
        public void IncrementDiscard_IncrementsDiscardCount()
        {
            // Arrange
            var stats = new PoolStatistics();

            // Act
            stats.IncrementDiscard();
            stats.IncrementDiscard();
            stats.IncrementDiscard();

            // Assert
            Assert.Equal(3, stats.DiscardCount);
        }

        [Fact]
        public void Reset_ResetsAllCounters()
        {
            // Arrange
            var stats = new PoolStatistics();
            stats.IncrementRent();
            stats.IncrementReturn();
            stats.IncrementMiss();
            stats.IncrementDiscard();

            // Act
            stats.Reset();

            // Assert
            Assert.Equal(0, stats.RentCount);
            Assert.Equal(0, stats.ReturnCount);
            Assert.Equal(0, stats.MissCount);
            Assert.Equal(0, stats.DiscardCount);
            Assert.Equal(0.0, stats.HitRate);
        }

        [Fact]
        public void MultipleOperations_TrackCorrectly()
        {
            // Arrange
            var stats = new PoolStatistics();

            // Act
            for (int i = 0; i < 100; i++)
            {
                stats.IncrementRent();
                if (i % 5 == 0)
                {
                    stats.IncrementMiss();
                }
                stats.IncrementReturn();
                if (i % 10 == 0)
                {
                    stats.IncrementDiscard();
                }
            }

            // Assert
            Assert.Equal(100, stats.RentCount);
            Assert.Equal(100, stats.ReturnCount);
            Assert.Equal(20, stats.MissCount); // Every 5th
            Assert.Equal(10, stats.DiscardCount); // Every 10th
            Assert.Equal(0.8, stats.HitRate, 4); // 80% hit rate
        }
    }
}
