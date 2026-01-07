using MLFramework.Quantization.Backends;
using MLFramework.Quantization.Backends.GPUBackend;
using Xunit;

namespace MLFramework.Tests.Quantization.Backends
{
    /// <summary>
    /// Tests for GPU backend with CUDA support.
    /// </summary>
    public class GPUBackendTests
    {
        [Fact]
        public void GPUBackend_IsAvailable_ReturnsBasedOnSystem()
        {
            // Arrange
            var backend = new GPUBackend();

            // Act
            var isAvailable = backend.IsAvailable();

            // Assert - Result depends on system, but should not throw
            Assert.NotNull(backend);
        }

        [Fact]
        public void GPUBackend_GetName_ReturnsCorrectName()
        {
            // Arrange
            var backend = new GPUBackend();

            // Act
            var name = backend.GetName();

            // Assert
            Assert.Equal("CUDA Tensor Cores", name);
        }

        [Fact]
        public void GPUBackend_GetCapabilities_ReturnsValidCapabilities()
        {
            // Arrange
            var backend = new GPUBackend();

            // Act
            var capabilities = backend.GetCapabilities();

            // Assert
            Assert.NotNull(capabilities);
        }

        [Fact]
        public void CUDAFeatureDetection_GetDeviceInfo_ReturnsValidString()
        {
            // Act
            var deviceInfo = CUDAFeatureDetection.GetDeviceInfo();

            // Assert
            Assert.NotNull(deviceInfo);
        }
    }
}
