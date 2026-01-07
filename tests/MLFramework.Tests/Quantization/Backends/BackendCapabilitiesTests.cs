using MLFramework.Quantization.Backends;
using Xunit;

namespace MLFramework.Tests.Quantization.Backends
{
    /// <summary>
    /// Tests for backend capabilities and capability flags.
    /// </summary>
    public class BackendCapabilitiesTests
    {
        [Fact]
        public void BackendCapabilityFlags_Int8MatMul_HasCorrectValue()
        {
            // Act & Assert
            Assert.Equal(1 << 0, (int)BackendCapabilityFlags.Int8MatMul);
        }

        [Fact]
        public void BackendCapabilityFlags_Int8Conv2D_HasCorrectValue()
        {
            // Act & Assert
            Assert.Equal(1 << 1, (int)BackendCapabilityFlags.Int8Conv2D);
        }

        [Fact]
        public void BackendCapabilityFlags_PerChannelQuantization_HasCorrectValue()
        {
            // Act & Assert
            Assert.Equal(1 << 2, (int)BackendCapabilityFlags.PerChannelQuantization);
        }

        [Fact]
        public void BackendCapabilityFlags_MixedPrecision_HasCorrectValue()
        {
            // Act & Assert
            Assert.Equal(1 << 3, (int)BackendCapabilityFlags.MixedPrecision);
        }

        [Fact]
        public void BackendCapabilityFlags_CombineWithOr()
        {
            // Arrange
            var flags = BackendCapabilityFlags.Int8MatMul | BackendCapabilityFlags.Int8Conv2D;

            // Act & Assert
            Assert.True((flags & BackendCapabilityFlags.Int8MatMul) != 0);
            Assert.True((flags & BackendCapabilityFlags.Int8Conv2D) != 0);
        }

        [Fact]
        public void BackendCapabilities_SupportsInt8MatMul_ReturnsTrueWhenFlagSet()
        {
            // Arrange
            var caps = new BackendCapabilities(
                BackendCapabilityFlags.Int8MatMul);

            // Act & Assert
            Assert.True(caps.SupportsInt8MatMul);
        }

        [Fact]
        public void BackendCapabilities_SupportsInt8MatMul_ReturnsFalseWhenFlagNotSet()
        {
            // Arrange
            var caps = new BackendCapabilities(BackendCapabilityFlags.None);

            // Act & Assert
            Assert.False(caps.SupportsInt8MatMul);
        }

        [Fact]
        public void BackendCapabilities_SupportsInt8Conv2D_ReturnsTrueWhenFlagSet()
        {
            // Arrange
            var caps = new BackendCapabilities(
                BackendCapabilityFlags.Int8Conv2D);

            // Act & Assert
            Assert.True(caps.SupportsInt8Conv2D);
        }

        [Fact]
        public void BackendCapabilities_SupportsPerChannelQuantization_ReturnsTrueWhenFlagSet()
        {
            // Arrange
            var caps = new BackendCapabilities(
                BackendCapabilityFlags.PerChannelQuantization);

            // Act & Assert
            Assert.True(caps.SupportsPerChannelQuantization);
        }

        [Fact]
        public void BackendCapabilities_SupportsMixedPrecision_ReturnsTrueWhenFlagSet()
        {
            // Arrange
            var caps = new BackendCapabilities(
                BackendCapabilityFlags.MixedPrecision);

            // Act & Assert
            Assert.True(caps.SupportsMixedPrecision);
        }

        [Fact]
        public void BackendCapabilities_MaxTensorSize_DefaultsToMaxValue()
        {
            // Arrange
            var caps = new BackendCapabilities(BackendCapabilityFlags.None);

            // Act & Assert
            Assert.Equal(long.MaxValue, caps.MaxTensorSize);
        }

        [Fact]
        public void BackendCapabilities_MinBatchSize_DefaultsToOne()
        {
            // Arrange
            var caps = new BackendCapabilities(BackendCapabilityFlags.None);

            // Act & Assert
            Assert.Equal(1, caps.MinBatchSize);
        }

        [Fact]
        public void BackendCapabilities_PreferredBatchSize_DefaultsTo32()
        {
            // Arrange
            var caps = new BackendCapabilities(BackendCapabilityFlags.None);

            // Act & Assert
            Assert.Equal(32, caps.PreferredBatchSize);
        }

        [Fact]
        public void BackendCapabilities_CustomValues_AreSetCorrectly()
        {
            // Arrange
            var caps = new BackendCapabilities(
                flags: BackendCapabilityFlags.Int8MatMul,
                maxTensorSize: 1024L * 1024 * 1024,
                minBatchSize: 4,
                preferredBatchSize: 64,
                maxThreads: 16);

            // Act & Assert
            Assert.Equal(1024L * 1024 * 1024, caps.MaxTensorSize);
            Assert.Equal(4, caps.MinBatchSize);
            Assert.Equal(64, caps.PreferredBatchSize);
            Assert.Equal(16, caps.MaxThreads);
        }

        [Fact]
        public void BackendCapabilities_ToString_ReturnsReadableString()
        {
            // Arrange
            var caps = new BackendCapabilities(
                BackendCapabilityFlags.Int8MatMul | BackendCapabilityFlags.Int8Conv2D);

            // Act
            var str = caps.ToString();

            // Assert
            Assert.Contains("Int8MatMul", str);
            Assert.Contains("Int8Conv2D", str);
        }
    }
}
