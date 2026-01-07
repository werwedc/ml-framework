using MLFramework.Quantization.Backends;
using Xunit;

namespace MLFramework.Tests.Quantization.Backends
{
    /// <summary>
    /// Tests for BackendFactory functionality.
    /// </summary>
    public class BackendFactoryTests
    {
        [Fact]
        public void CreateDefault_ReturnsCPUBackend_WhenNoOtherAvailable()
        {
            // Act
            var backend = BackendFactory.CreateDefault();

            // Assert
            Assert.NotNull(backend);
            Assert.True(backend.IsAvailable());
        }

        [Fact]
        public void CreateCPU_ReturnsCPUBackend()
        {
            // Act
            var backend = BackendFactory.Create("CPU");

            // Assert
            Assert.NotNull(backend);
            Assert.Equal("CPU", backend.GetName());
            Assert.True(backend.IsAvailable());
        }

        [Fact]
        public void Create_InvalidBackendName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => BackendFactory.Create("InvalidBackend"));
        }

        [Fact]
        public void GetAvailableBackends_ContainsCPU()
        {
            // Act
            var backends = BackendFactory.GetAvailableBackends();

            // Assert
            Assert.NotNull(backends);
            Assert.Contains("CPU", backends);
        }

        [Fact]
        public void GetAvailableBackends_ReturnsNonEmptyList()
        {
            // Act
            var backends = BackendFactory.GetAvailableBackends();

            // Assert
            Assert.NotNull(backends);
            Assert.NotEmpty(backends);
        }

        [Fact]
        public void SetPreferredBackend_ValidName_SetsPreference()
        {
            // Arrange
            BackendFactory.SetPreferredBackend("CPU");

            // Act
            var backend = BackendFactory.CreateDefault();

            // Assert
            Assert.Equal("CPU", backend.GetName());
        }

        [Fact]
        public void SetPreferredBackend_InvalidName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => BackendFactory.SetPreferredBackend("InvalidBackend"));
        }

        [Fact]
        public void ClearCache_ClearsBackendCache()
        {
            // Arrange
            BackendFactory.Create("CPU");
            BackendFactory.ClearCache();

            // Act - This should create a new instance
            var backend = BackendFactory.Create("CPU");

            // Assert
            Assert.NotNull(backend);
        }

        [Fact]
        public void Create_CachesBackendInstances()
        {
            // Arrange
            var backend1 = BackendFactory.Create("CPU");
            var backend2 = BackendFactory.Create("CPU");

            // Assert - Same instance should be returned from cache
            Assert.Same(backend1, backend2);
        }

        [Fact]
        public void Create_CaseInsensitive_BackendName()
        {
            // Act
            var backend1 = BackendFactory.Create("CPU");
            var backend2 = BackendFactory.Create("cpu");
            var backend3 = BackendFactory.Create("Cpu");

            // Assert
            Assert.NotNull(backend1);
            Assert.NotNull(backend2);
            Assert.NotNull(backend3);
            Assert.Same(backend1, backend2);
        }
    }
}
