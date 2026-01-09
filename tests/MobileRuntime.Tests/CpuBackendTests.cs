using Microsoft.VisualStudio.TestTools.UnitTesting;
using FluentAssertions;

namespace MobileRuntime.Tests
{
    [TestClass]
    public class CpuBackendTests
    {
        [TestMethod]
        public void BackendType_Enum_HasExpectedValues()
        {
            // Act & Assert
            Assert.IsNotNull(BackendType.Cpu);
            Assert.IsNotNull(BackendType.Metal);
            Assert.IsNotNull(BackendType.Vulkan);
        }

        [TestMethod]
        public void BackendType_Default_IsCpu()
        {
            // Arrange
            var backendType = BackendType.Cpu;

            // Assert
            backendType.Should().Be(BackendType.Cpu);
        }
    }
}
