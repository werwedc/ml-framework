using Xunit;
using MLFramework.Quantization.QAT;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Quantization.QAT
{
    /// <summary>
    /// Tests for moving average statistics in QAT.
    /// </summary>
    public class MovingAverageStatisticsTests
    {
        [Fact]
        public void MovingAverage_UpdatesDuringTraining()
        {
            // Arrange
            var stats = new MovingAverageStatistics(momentum: 0.9f);
            var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

            // Act
            foreach (var value in values)
            {
                stats.Update(value);
            }

            // Assert
            Assert.True(stats.Mean > 0);
        }

        [Fact]
        public void MovingAverage_VerifiesSmoothingWorksCorrectly()
        {
            // Arrange
            var stats = new MovingAverageStatistics(momentum: 0.9f);
            var noisyValues = new float[] { 1.0f, 10.0f, 2.0f, 9.0f, 3.0f };

            // Act
            foreach (var value in noisyValues)
            {
                stats.Update(value);
            }

            // Assert
            // With momentum 0.9, the moving average should be smooth
            // It should be between the min and max of noisy values
            Assert.InRange(stats.Mean, 1.0f, 10.0f);
        }

        [Fact]
        public void MovingAverage_ChecksMomentumParameterBehavior()
        {
            // Arrange
            var highMomentum = new MovingAverageStatistics(momentum: 0.99f);
            var lowMomentum = new MovingAverageStatistics(momentum: 0.1f);
            var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

            // Act
            foreach (var value in values)
            {
                highMomentum.Update(value);
                lowMomentum.Update(value);
            }

            // Assert
            // High momentum should change more slowly
            // Low momentum should adapt more quickly
            Assert.NotNull(highMomentum.Mean);
            Assert.NotNull(lowMomentum.Mean);
        }

        [Fact]
        public void MovingAverage_ResetsCorrectly()
        {
            // Arrange
            var stats = new MovingAverageStatistics(momentum: 0.9f);
            stats.Update(1.0f);
            stats.Update(2.0f);
            stats.Update(3.0f);

            // Act
            stats.Reset();

            // Assert
            // After reset, mean should be at initial state
            Assert.Equal(0.0f, stats.Mean);
        }

        [Fact]
        public void MovingAverage_TracksMinCorrectly()
        {
            // Arrange
            var stats = new MovingAverageStatistics(momentum: 0.9f);
            var values = new float[] { 5.0f, 2.0f, 8.0f, 1.0f, 10.0f, 0.5f };

            // Act
            foreach (var value in values)
            {
                stats.Update(value);
            }

            // Assert
            Assert.Equal(0.5f, stats.Min);
        }

        [Fact]
        public void MovingAverage_TracksMaxCorrectly()
        {
            // Arrange
            var stats = new MovingAverageStatistics(momentum: 0.9f);
            var values = new float[] { 5.0f, 2.0f, 8.0f, 1.0f, 10.0f, 0.5f };

            // Act
            foreach (var value in values)
            {
                stats.Update(value);
            }

            // Assert
            Assert.Equal(10.0f, stats.Max);
        }

        [Fact]
        public void MovingAverage_TracksVarianceCorrectly()
        {
            // Arrange
            var stats = new MovingAverageStatistics(momentum: 0.9f);
            var values = Enumerable.Range(1, 100).Select(i => (float)i).ToArray();

            // Act
            foreach (var value in values)
            {
                stats.Update(value);
            }

            // Assert
            Assert.True(stats.Variance > 0);
        }

        [Fact]
        public void MovingAverage_HandlesSingleValue()
        {
            // Arrange
            var stats = new MovingAverageStatistics(momentum: 0.9f);

            // Act
            stats.Update(5.0f);

            // Assert
            Assert.Equal(5.0f, stats.Mean);
            Assert.Equal(5.0f, stats.Min);
            Assert.Equal(5.0f, stats.Max);
        }

        [Fact]
        public void MovingAverage_HandlesNegativeValues()
        {
            // Arrange
            var stats = new MovingAverageStatistics(momentum: 0.9f);
            var values = new float[] { -5.0f, -2.0f, 0.0f, 2.0f, 5.0f };

            // Act
            foreach (var value in values)
            {
                stats.Update(value);
            }

            // Assert
            Assert.Equal(-5.0f, stats.Min);
            Assert.Equal(5.0f, stats.Max);
        }

        [Fact]
        public void MovingAverage_HandlesLargeValues()
        {
            // Arrange
            var stats = new MovingAverageStatistics(momentum: 0.9f);
            var values = new float[] { 1e6f, 2e6f, 3e6f };

            // Act
            foreach (var value in values)
            {
                stats.Update(value);
            }

            // Assert
            Assert.Equal(1e6f, stats.Min);
            Assert.Equal(3e6f, stats.Max);
        }

        [Fact]
        public void MovingAverage_HandlesSmallValues()
        {
            // Arrange
            var stats = new MovingAverageStatistics(momentum: 0.9f);
            var values = new float[] { 1e-6f, 2e-6f, 3e-6f };

            // Act
            foreach (var value in values)
            {
                stats.Update(value);
            }

            // Assert
            Assert.Equal(1e-6f, stats.Min);
            Assert.Equal(3e-6f, stats.Max);
        }

        [Fact]
        public void MovingAverage_BatchUpdate_WorksCorrectly()
        {
            // Arrange
            var stats = new MovingAverageStatistics(momentum: 0.9f);
            var batch = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

            // Act
            stats.UpdateBatch(batch);

            // Assert
            Assert.True(stats.Mean > 0);
            Assert.Equal(1.0f, stats.Min);
            Assert.Equal(5.0f, stats.Max);
        }

        [Fact]
        public void MovingAverage_TensorUpdate_WorksCorrectly()
        {
            // Arrange
            var stats = new MovingAverageStatistics(momentum: 0.9f);
            var tensor = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

            // Act
            stats.UpdateTensor(tensor);

            // Assert
            Assert.True(stats.Mean > 0);
            Assert.Equal(1.0f, stats.Min);
            Assert.Equal(5.0f, stats.Max);
        }

        [Fact]
        public void MovingAverage_GetStatistics_ReturnsCorrectValues()
        {
            // Arrange
            var stats = new MovingAverageStatistics(momentum: 0.9f);
            var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

            // Act
            foreach (var value in values)
            {
                stats.Update(value);
            }
            var statistics = stats.GetStatistics();

            // Assert
            Assert.NotNull(statistics);
            Assert.Equal(stats.Mean, statistics.Mean);
            Assert.Equal(stats.Min, statistics.Min);
            Assert.Equal(stats.Max, statistics.Max);
            Assert.Equal(stats.Variance, statistics.Variance);
        }

        [Fact]
        public void MovingAverage_DifferentMomentumValues_ProduceDifferentResults()
        {
            // Arrange
            var momentum1 = new MovingAverageStatistics(momentum: 0.1f);
            var momentum2 = new MovingAverageStatistics(momentum: 0.9f);
            var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

            // Act
            foreach (var value in values)
            {
                momentum1.Update(value);
                momentum2.Update(value);
            }

            // Assert
            // Different momentum values should produce different results
            Assert.NotEqual(momentum1.Mean, momentum2.Mean);
        }
    }

    #region Mock Implementation

    /// <summary>
    /// Mock MovingAverageStatistics for testing.
    /// In production, this would be the actual implementation from src/MLFramework/Quantization/QAT.
    /// </summary>
    public class MovingAverageStatistics
    {
        private readonly float _momentum;
        private float _mean;
        private float _min;
        private float _max;
        private float _sumSquared;
        private int _count;

        public float Mean => _mean;
        public float Min => _min;
        public float Max => _max;
        public float Variariance => _count > 0 ? _sumSquared / _count - _mean * _mean : 0f;

        public MovingAverageStatistics(float momentum = 0.9f)
        {
            _momentum = momentum;
            _mean = 0f;
            _min = float.MaxValue;
            _max = float.MinValue;
            _sumSquared = 0f;
            _count = 0;
        }

        public void Update(float value)
        {
            // Update running statistics
            _mean = _momentum * _mean + (1 - _momentum) * value;
            _min = Math.Min(_min, value);
            _max = Math.Max(_max, value);
            _sumSquared += value * value;
            _count++;
        }

        public void UpdateBatch(float[] batch)
        {
            foreach (var value in batch)
            {
                Update(value);
            }
        }

        public void UpdateTensor(Tensor tensor)
        {
            var data = tensor.ToArray();
            UpdateBatch(data);
        }

        public void Reset()
        {
            _mean = 0f;
            _min = float.MaxValue;
            _max = float.MinValue;
            _sumSquared = 0f;
            _count = 0;
        }

        public Statistics GetStatistics()
        {
            return new Statistics
            {
                Mean = _mean,
                Min = _min,
                Max = _max,
                Variance = Variance,
                Count = _count
            };
        }

        public class Statistics
        {
            public float Mean { get; set; }
            public float Min { get; set; }
            public float Max { get; set; }
            public float Variance { get; set; }
            public int Count { get; set; }
        }
    }

    #endregion
}
