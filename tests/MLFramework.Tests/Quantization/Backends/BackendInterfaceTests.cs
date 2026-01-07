using MLFramework.Quantization.Backends;
using MLFramework.Quantization.DataStructures;
using Xunit;

namespace MLFramework.Tests.Quantization.Backends
{
    /// <summary>
    /// Tests for IQuantizationBackend interface compliance.
    /// </summary>
    public class BackendInterfaceTests
    {
        [Fact]
        public void AllBackends_ImplementInterfaceCorrectly()
        {
            // Arrange & Act
            var cpuBackend = new Backends.CPUBackend.CPUBackend();
            var x86Backend = new Backends.x86Backend.x86Backend();
            var armBackend = new Backends.ARMBackend.ARMBackend();
            var gpuBackend = new Backends.GPUBackend.GPUBackend();

            // Assert
            Assert.IsAssignableFrom<IQuantizationBackend>(cpuBackend);
            Assert.IsAssignableFrom<IQuantizationBackend>(x86Backend);
            Assert.IsAssignableFrom<IQuantizationBackend>(armBackend);
            Assert.IsAssignableFrom<IQuantizationBackend>(gpuBackend);
        }

        [Fact]
        public void CPUBackend_IsAvailable_ReturnsTrue()
        {
            // Arrange
            var backend = new Backends.CPUBackend.CPUBackend();

            // Act
            var isAvailable = backend.IsAvailable();

            // Assert
            Assert.True(isAvailable);
        }

        [Fact]
        public void CPUBackend_GetName_ReturnsCorrectName()
        {
            // Arrange
            var backend = new Backends.CPUBackend.CPUBackend();

            // Act
            var name = backend.GetName();

            // Assert
            Assert.Equal("CPU", name);
        }

        [Fact]
        public void x86Backend_GetName_ReturnsCorrectName()
        {
            // Arrange
            var backend = new Backends.x86Backend.x86Backend();

            // Act
            var name = backend.GetName();

            // Assert
            Assert.Equal("Intel oneDNN", name);
        }

        [Fact]
        public void ARMBackend_GetName_ReturnsCorrectName()
        {
            // Arrange
            var backend = new Backends.ARMBackend.ARMBackend();

            // Act
            var name = backend.GetName();

            // Assert
            Assert.Equal("ARM NEON", name);
        }

        [Fact]
        public void GPUBackend_GetName_ReturnsCorrectName()
        {
            // Arrange
            var backend = new Backends.GPUBackend.GPUBackend();

            // Act
            var name = backend.GetName();

            // Assert
            Assert.Equal("CUDA Tensor Cores", name);
        }

        [Fact]
        public void AllBackends_GetCapabilities_ReturnsValidCapabilities()
        {
            // Arrange
            var cpuBackend = new Backends.CPUBackend.CPUBackend();
            var x86Backend = new Backends.x86Backend.x86Backend();
            var armBackend = new Backends.ARMBackend.ARMBackend();
            var gpuBackend = new Backends.GPUBackend.GPUBackend();

            // Act
            var cpuCaps = cpuBackend.GetCapabilities();
            var x86Caps = x86Backend.GetCapabilities();
            var armCaps = armBackend.GetCapabilities();
            var gpuCaps = gpuBackend.GetCapabilities();

            // Assert
            Assert.True(cpuCaps.SupportsInt8MatMul);
            Assert.True(cpuCaps.SupportsInt8Conv2D);
            Assert.True(cpuCaps.SupportsPerChannelQuantization);

            // x86, ARM, and GPU backends should also have capabilities
            Assert.True(x86Caps.SupportsInt8MatMul);
            Assert.True(armCaps.SupportsInt8MatMul);
            Assert.True(gpuCaps.SupportsInt8MatMul);
        }
    }
}
