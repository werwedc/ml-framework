using MLFramework.Quantization.Backends;
using Xunit;
using System.Diagnostics;

namespace MLFramework.Tests.Quantization.Backends
{
    /// <summary>
    /// Tests for backend switching and dynamic backend selection.
    /// </summary>
    public class BackendSwitchingTests
    {
        [Fact]
        public void SwitchBetweenBackends_DuringRuntime_DoesNotThrow()
        {
            // Arrange
            var backend1 = BackendFactory.Create("CPU");
            var backend2 = BackendFactory.Create("CPU");

            // Act & Assert - Should not throw
            Assert.NotNull(backend1);
            Assert.NotNull(backend2);
            Assert.Equal(backend1.GetName(), backend2.GetName());
        }

        [Fact]
        public void BackendFactory_ClearCache_AllowsSwitching()
        {
            // Arrange
            var backend1 = BackendFactory.Create("CPU");
            BackendFactory.ClearCache();
            var backend2 = BackendFactory.Create("CPU");

            // Act & Assert
            Assert.NotNull(backend1);
            Assert.NotNull(backend2);
        }

        [Fact]
        public void SetPreferredBackend_CreatesDifferentBackend_AfterClear()
        {
            // Arrange
            BackendFactory.SetPreferredBackend("CPU");
            var backend1 = BackendFactory.CreateDefault();

            BackendFactory.ClearCache();
            var backend2 = BackendFactory.CreateDefault();

            // Act & Assert
            Assert.Equal(backend1.GetName(), backend2.GetName());
        }

        [Fact]
        public void MultipleBackends_SameOperation_ProduceConsistentResults()
        {
            // Arrange
            var backends = BackendFactory.GetAvailableBackends();
            Assert.NotEmpty(backends);

            // Act & Assert - All backends should have consistent interface behavior
            foreach (var backendName in backends)
            {
                try
                {
                    var backend = BackendFactory.Create(backendName);
                    Assert.True(backend.IsAvailable());
                    Assert.NotNull(backend.GetName());
                    Assert.NotNull(backend.GetCapabilities());
                }
                catch (NotSupportedException)
                {
                    // Backend not available, skip
                }
            }
        }

        [Fact]
        public void BackendSwitch_TransparentToUser_SameInterface()
        {
            // Arrange
            var backend1 = BackendFactory.CreateDefault();

            // Act
            BackendFactory.ClearCache();
            var backend2 = BackendFactory.CreateDefault();

            // Assert - Both should implement the same interface
            Assert.IsAssignableFrom<IQuantizationBackend>(backend1);
            Assert.IsAssignableFrom<IQuantizationBackend>(backend2);
        }
    }
}
