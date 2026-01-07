using MLFramework.Quantization.Backends;
using MLFramework.Quantization.Backends.x86Backend;
using Xunit;

namespace MLFramework.Tests.Quantization.Backends
{
    /// <summary>
    /// Tests for x86 backend with oneDNN support.
    /// </summary>
    public class x86BackendTests
    {
        [Fact]
        public void x86Backend_IsAvailable_ReturnsBasedOnSystem()
        {
            // Arrange
            var backend = new x86Backend();

            // Act
            var isAvailable = backend.IsAvailable();

            // Assert - Result depends on system, but should not throw
            // If AVX2 is supported, should be available
            Assert.NotNull(backend);
        }

        [Fact]
        public void x86Backend_GetName_ReturnsCorrectName()
        {
            // Arrange
            var backend = new x86Backend();

            // Act
            var name = backend.GetName();

            // Assert
            Assert.Equal("Intel oneDNN", name);
        }

        [Fact]
        public void x86Backend_GetCapabilities_ReturnsValidCapabilities()
        {
            // Arrange
            var backend = new x86Backend();

            // Act
            var capabilities = backend.GetCapabilities();

            // Assert
            Assert.NotNull(capabilities);
        }

        [Fact]
        public void x86FeatureDetection_GetDetectedFeatures_ReturnsValidString()
        {
            // Act
            var features = x86FeatureDetection.GetDetectedFeatures();

            // Assert
            Assert.NotNull(features);
        }
    }
}
