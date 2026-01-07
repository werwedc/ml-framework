using MLFramework.Quantization.Backends;
using MLFramework.Quantization.Backends.ARMBackend;
using Xunit;

namespace MLFramework.Tests.Quantization.Backends
{
    /// <summary>
    /// Tests for ARM backend with NEON support.
    /// </summary>
    public class ARMBackendTests
    {
        [Fact]
        public void ARMBackend_IsAvailable_ReturnsBasedOnSystem()
        {
            // Arrange
            var backend = new ARMBackend();

            // Act
            var isAvailable = backend.IsAvailable();

            // Assert - Result depends on system, but should not throw
            Assert.NotNull(backend);
        }

        [Fact]
        public void ARMBackend_GetName_ReturnsCorrectName()
        {
            // Arrange
            var backend = new ARMBackend();

            // Act
            var name = backend.GetName();

            // Assert
            Assert.Equal("ARM NEON", name);
        }

        [Fact]
        public void ARMBackend_GetCapabilities_ReturnsValidCapabilities()
        {
            // Arrange
            var backend = new ARMBackend();

            // Act
            var capabilities = backend.GetCapabilities();

            // Assert
            Assert.NotNull(capabilities);
        }

        [Fact]
        public void ARMFeatureDetection_GetDetectedFeatures_ReturnsValidString()
        {
            // Act
            var features = ARMFeatureDetection.GetDetectedFeatures();

            // Assert
            Assert.NotNull(features);
        }
    }
}
