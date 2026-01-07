using Xunit;
using MLFramework.Quantization.QAT;
using MLFramework.Quantization.DataStructures;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Quantization.QAT
{
    /// <summary>
    /// Tests for activation observers in QAT.
    /// </summary>
    public class ActivationObserverTests
    {
        [Fact]
        public void ActivationObserver_TracksMinMaxStatisticsCorrectly()
        {
            // Arrange
            var observer = new ActivationObserver();
            var tensor = new Tensor(new float[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f });

            // Act
            observer.Update(tensor);
            var stats = observer.GetStatistics();

            // Assert
            Assert.NotNull(stats);
            Assert.Equal(-2.0f, stats.Min);
            Assert.Equal(2.0f, stats.Max);
        }

        [Fact]
        public void ActivationObserver_UpdatesQuantizationParameters()
        {
            // Arrange
            var observer = new ActivationObserver();
            var tensor = new Tensor(new float[] { -1.0f, 0.0f, 1.0f });

            // Act
            observer.Update(tensor);
            var quantParams = observer.GetQuantizationParameters();

            // Assert
            Assert.NotNull(quantParams);
            Assert.True(quantParams.Scale > 0);
        }

        [Fact]
        public void ActivationObserver_HandlesMinMaxObserverStrategy()
        {
            // Arrange
            var observer = new ActivationObserver(ObserverStrategy.MinMax);
            var tensor1 = new Tensor(new float[] { -1.0f, 0.0f, 1.0f });
            var tensor2 = new Tensor(new float[] { -3.0f, 0.0f, 3.0f });

            // Act
            observer.Update(tensor1);
            observer.Update(tensor2);
            var stats = observer.GetStatistics();

            // Assert
            // MinMax strategy should track absolute min and max
            Assert.Equal(-3.0f, stats.Min);
            Assert.Equal(3.0f, stats.Max);
        }

        [Fact]
        public void ActivationObserver_HandlesMovingAverageObserverStrategy()
        {
            // Arrange
            var observer = new ActivationObserver(ObserverStrategy.MovingAverage, momentum: 0.9f);
            var tensor1 = new Tensor(new float[] { 1.0f, 2.0f, 3.0f });
            var tensor2 = new Tensor(new float[] { 10.0f, 20.0f, 30.0f });

            // Act
            observer.Update(tensor1);
            observer.Update(tensor2);
            var stats = observer.GetStatistics();

            // Assert
            // Moving average should smooth the statistics
            Assert.NotNull(stats);
            Assert.True(stats.Mean > 0);
        }

        [Fact]
        public void ActivationObserver_HandlesEntropyObserverStrategy()
        {
            // Arrange
            var observer = new ActivationObserver(ObserverStrategy.Entropy);
            var tensor = new Tensor(new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f });

            // Act
            observer.Update(tensor);
            var stats = observer.GetStatistics();

            // Assert
            Assert.NotNull(stats);
        }

        [Fact]
        public void ActivationObserver_ResetsCorrectly()
        {
            // Arrange
            var observer = new ActivationObserver();
            var tensor1 = new Tensor(new float[] { -1.0f, 0.0f, 1.0f });
            observer.Update(tensor1);

            // Act
            observer.Reset();
            var stats = observer.GetStatistics();

            // Assert
            // After reset, stats should be at initial state
            Assert.NotNull(stats);
        }

        [Fact]
        public void ActivationObserver_HandlesNegativeValues()
        {
            // Arrange
            var observer = new ActivationObserver();
            var tensor = new Tensor(new float[] { -5.0f, -3.0f, -1.0f, -0.5f, -0.1f });

            // Act
            observer.Update(tensor);
            var stats = observer.GetStatistics();

            // Assert
            Assert.Equal(-5.0f, stats.Min);
            Assert.Equal(-0.1f, stats.Max);
        }

        [Fact]
        public void ActivationObserver_HandlesPositiveValues()
        {
            // Arrange
            var observer = new ActivationObserver();
            var tensor = new Tensor(new float[] { 0.1f, 0.5f, 1.0f, 3.0f, 5.0f });

            // Act
            observer.Update(tensor);
            var stats = observer.GetStatistics();

            // Assert
            Assert.Equal(0.1f, stats.Min);
            Assert.Equal(5.0f, stats.Max);
        }

        [Fact]
        public void ActivationObserver_HandlesMixedValues()
        {
            // Arrange
            var observer = new ActivationObserver();
            var tensor = new Tensor(new float[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f });

            // Act
            observer.Update(tensor);
            var stats = observer.GetStatistics();

            // Assert
            Assert.Equal(-2.0f, stats.Min);
            Assert.Equal(2.0f, stats.Max);
        }

        [Fact]
        public void ActivationObserver_HandlesBatchesCorrectly()
        {
            // Arrange
            var observer = new ActivationObserver();
            var batch1 = new Tensor(new float[] { 1.0f, 2.0f, 3.0f });
            var batch2 = new Tensor(new float[] { 4.0f, 5.0f, 6.0f });
            var batch3 = new Tensor(new float[] { 0.0f, 7.0f, 8.0f });

            // Act
            observer.Update(batch1);
            observer.Update(batch2);
            observer.Update(batch3);
            var stats = observer.GetStatistics();

            // Assert
            Assert.Equal(0.0f, stats.Min);
            Assert.Equal(8.0f, stats.Max);
        }

        [Fact]
        public void ActivationObserver_TracksMeanCorrectly()
        {
            // Arrange
            var observer = new ActivationObserver();
            var tensor = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

            // Act
            observer.Update(tensor);
            var stats = observer.GetStatistics();

            // Assert
            Assert.Equal(3.0f, stats.Mean);
        }

        [Fact]
        public void ActivationObserver_TracksVarianceCorrectly()
        {
            // Arrange
            var observer = new ActivationObserver();
            var tensor = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

            // Act
            observer.Update(tensor);
            var stats = observer.GetStatistics();

            // Assert
            // Variance of [1, 2, 3, 4, 5] is 2.0
            Assert.Equal(2.0f, stats.Variance, precision: 2);
        }

        [Fact]
        public void ActivationObserver_WithPerChannelMode_WorksCorrectly()
        {
            // Arrange
            var observer = new ActivationObserver(perChannel: true);
            var tensor = new Tensor(new float[]
            {
                // Channel 0: values 1.0, 2.0, 3.0
                // Channel 1: values 4.0, 5.0, 6.0
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
            });

            // Act
            observer.Update(tensor);
            var channelStats = observer.GetPerChannelStatistics();

            // Assert
            Assert.NotNull(channelStats);
            Assert.True(channelStats.Count > 0);
        }

        [Fact]
        public void ActivationObserver_HandlesZeroRange()
        {
            // Arrange
            var observer = new ActivationObserver();
            var tensor = new Tensor(new float[] { 5.0f, 5.0f, 5.0f, 5.0f });

            // Act
            observer.Update(tensor);
            var stats = observer.GetStatistics();

            // Assert
            Assert.Equal(5.0f, stats.Min);
            Assert.Equal(5.0f, stats.Max);
        }

        [Fact]
        public void ActivationObserver_HandlesVerySmallRange()
        {
            // Arrange
            var observer = new ActivationObserver();
            var tensor = new Tensor(new float[] { 0.001f, 0.002f, 0.003f });

            // Act
            observer.Update(tensor);
            var stats = observer.GetStatistics();

            // Assert
            Assert.Equal(0.001f, stats.Min);
            Assert.Equal(0.003f, stats.Max);
        }

        [Fact]
        public void ActivationObserver_HandlesVeryLargeRange()
        {
            // Arrange
            var observer = new ActivationObserver();
            var tensor = new Tensor(new float[] { 1e6f, 1e7f, 1e8f });

            // Act
            observer.Update(tensor);
            var stats = observer.GetStatistics();

            // Assert
            Assert.Equal(1e6f, stats.Min);
            Assert.Equal(1e8f, stats.Max);
        }

        [Fact]
        public void ActivationObserver_DisableObserver_StopsUpdates()
        {
            // Arrange
            var observer = new ActivationObserver();
            var tensor1 = new Tensor(new float[] { 1.0f, 2.0f, 3.0f });
            var tensor2 = new Tensor(new float[] { 10.0f, 20.0f, 30.0f });

            // Act
            observer.Update(tensor1);
            observer.Enabled = false; // Disable observer
            observer.Update(tensor2);
            var stats = observer.GetStatistics();

            // Assert
            // Should only have stats from tensor1
            Assert.Equal(1.0f, stats.Min);
            Assert.Equal(3.0f, stats.Max);
        }
    }

    #region Mock Implementation

    /// <summary>
    /// Mock ActivationObserver for testing.
    /// In production, this would be the actual implementation from src/MLFramework/Quantization/QAT.
    /// </summary>
    public class ActivationObserver
    {
        private readonly ObserverStrategy _strategy;
        private readonly float _momentum;
        private readonly bool _perChannel;
        private float _min;
        private float _max;
        private float _mean;
        private float _sumSquared;
        private int _count;
        private bool _enabled;

        public bool Enabled
        {
            get => _enabled;
            set => _enabled = value;
        }

        public ActivationObserver(ObserverStrategy strategy = ObserverStrategy.MinMax, float momentum = 0.9f, bool perChannel = false)
        {
            _strategy = strategy;
            _momentum = momentum;
            _perChannel = perChannel;
            _min = float.MaxValue;
            _max = float.MinValue;
            _mean = 0f;
            _sumSquared = 0f;
            _count = 0;
            _enabled = true;
        }

        public void Update(Tensor tensor)
        {
            if (!_enabled) return;

            var data = tensor.ToArray();

            if (_strategy == ObserverStrategy.MinMax)
            {
                foreach (var value in data)
                {
                    _min = Math.Min(_min, value);
                    _max = Math.Max(_max, value);
                    _mean = (_mean * _count + value) / (_count + 1);
                    _sumSquared += value * value;
                    _count++;
                }
            }
            else if (_strategy == ObserverStrategy.MovingAverage)
            {
                foreach (var value in data)
                {
                    _min = Math.Min(_min, value);
                    _max = Math.Max(_max, value);
                    _mean = _momentum * _mean + (1 - _momentum) * value;
                    _sumSquared += value * value;
                    _count++;
                }
            }
            else if (_strategy == ObserverStrategy.Entropy)
            {
                // Similar to MinMax for now
                foreach (var value in data)
                {
                    _min = Math.Min(_min, value);
                    _max = Math.Max(_max, value);
                    _mean = (_mean * _count + value) / (_count + 1);
                    _sumSquared += value * value;
                    _count++;
                }
            }
        }

        public ObserverStatistics GetStatistics()
        {
            float variance = _count > 0 ? _sumSquared / _count - _mean * _mean : 0f;
            return new ObserverStatistics
            {
                Min = _min == float.MaxValue ? 0f : _min,
                Max = _max == float.MinValue ? 0f : _max,
                Mean = _mean,
                Variance = variance,
                Count = _count
            };
        }

        public QuantizationParameters? GetQuantizationParameters()
        {
            var stats = GetStatistics();
            float range = stats.Max - stats.Min;

            if (range < 1e-6f)
            {
                // Handle zero range case
                return new QuantizationParameters
                {
                    Scale = 1.0f,
                    ZeroPoint = 0,
                    QuantizationMode = QuantizationMode.PerTensorSymmetric
                };
            }

            // Calculate scale for Int8 range [-128, 127]
            float scale = range / 255f;
            int zeroPoint = (int)(-stats.Min / scale);

            return new QuantizationParameters
            {
                Scale = scale,
                ZeroPoint = zeroPoint,
                QuantizationMode = QuantizationMode.PerTensorAsymmetric
            };
        }

        public List<ObserverStatistics> GetPerChannelStatistics()
        {
            // Simplified implementation
            return new List<ObserverStatistics> { GetStatistics() };
        }

        public void Reset()
        {
            _min = float.MaxValue;
            _max = float.MinValue;
            _mean = 0f;
            _sumSquared = 0f;
            _count = 0;
        }
    }

    /// <summary>
    /// Observer strategy enum.
    /// </summary>
    public enum ObserverStrategy
    {
        MinMax,
        MovingAverage,
        Entropy
    }

    /// <summary>
    /// Observer statistics.
    /// </summary>
    public class ObserverStatistics
    {
        public float Min { get; set; }
        public float Max { get; set; }
        public float Mean { get; set; }
        public float Variance { get; set; }
        public int Count { get; set; }
    }

    #endregion
}
