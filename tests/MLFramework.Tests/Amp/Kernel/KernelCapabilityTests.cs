using MLFramework.Amp;
using MLFramework.Core;
using Xunit;

namespace MLFramework.Tests.Amp.Kernel
{
    public class KernelCapabilityTests
    {
        [Fact]
        public void Constructor_SetsAllProperties()
        {
            // Arrange & Act
            var capability = new KernelCapability(
                KernelDtype.Float16,
                true,
                true,
                2.5f,
                1.8f);

            // Assert
            Assert.Equal(KernelDtype.Float16, capability.Dtype);
            Assert.True(capability.IsAvailable);
            Assert.True(capability.SupportsTensorCores);
            Assert.Equal(2.5f, capability.PerformanceFactor);
            Assert.Equal(1.8f, capability.MemoryEfficiency);
        }

        [Fact]
        public void CreateFloat32_CreatesCorrectCapability()
        {
            // Act
            var capability = KernelCapability.CreateFloat32(false);

            // Assert
            Assert.Equal(KernelDtype.Float32, capability.Dtype);
            Assert.True(capability.IsAvailable);
            Assert.False(capability.SupportsTensorCores);
            Assert.Equal(1.0f, capability.PerformanceFactor);
            Assert.Equal(1.0f, capability.MemoryEfficiency);
        }

        [Fact]
        public void CreateFloat32_WithTensorCores_SetsSupportsTensorCores()
        {
            // Act
            var capability = KernelCapability.CreateFloat32(true);

            // Assert
            Assert.True(capability.SupportsTensorCores);
        }

        [Fact]
        public void CreateFloat16_CreatesCorrectCapability()
        {
            // Act
            var capability = KernelCapability.CreateFloat16(false);

            // Assert
            Assert.Equal(KernelDtype.Float16, capability.Dtype);
            Assert.True(capability.IsAvailable);
            Assert.False(capability.SupportsTensorCores);
            Assert.Equal(2.0f, capability.PerformanceFactor);
            Assert.Equal(2.0f, capability.MemoryEfficiency);
        }

        [Fact]
        public void CreateFloat16_WithTensorCores_SetsSupportsTensorCores()
        {
            // Act
            var capability = KernelCapability.CreateFloat16(true);

            // Assert
            Assert.True(capability.SupportsTensorCores);
        }

        [Fact]
        public void CreateBFloat16_CreatesCorrectCapability()
        {
            // Act
            var capability = KernelCapability.CreateBFloat16(false);

            // Assert
            Assert.Equal(KernelDtype.BFloat16, capability.Dtype);
            Assert.True(capability.IsAvailable);
            Assert.False(capability.SupportsTensorCores);
            Assert.Equal(2.0f, capability.PerformanceFactor);
            Assert.Equal(2.0f, capability.MemoryEfficiency);
        }

        [Fact]
        public void CreateBFloat16_WithTensorCores_SetsSupportsTensorCores()
        {
            // Act
            var capability = KernelCapability.CreateBFloat16(true);

            // Assert
            Assert.True(capability.SupportsTensorCores);
        }

        [Fact]
        public void ToString_ReturnsFormattedString()
        {
            // Arrange
            var capability = KernelCapability.CreateFloat16(true);

            // Act
            var result = capability.ToString();

            // Assert
            Assert.Contains("Dtype=Float16", result);
            Assert.Contains("Available=True", result);
            Assert.Contains("TensorCores=True", result);
        }
    }
}
