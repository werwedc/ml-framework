using System;
using Xunit;
using MLFramework.Communication.Optimizations;
using MLFramework.Communication;

namespace MLFramework.Tests.Communication.Optimizations
{
    public class GPUDirectManagerTests
    {
        [Fact]
        public void IsSupported_ReturnsBoolean()
        {
            var isSupported = GPUDirectManager.IsSupported;
            Assert.IsType<bool>(isSupported);
        }

        [Fact]
        public void EnableGPUDirect_NotSupported_ThrowsException()
        {
            if (GPUDirectManager.IsSupported)
            {
                // Skip test if GPU-direct is supported
                return;
            }

            var backend = new MockCommunicationBackend(4);

            Assert.Throws<CommunicationException>(() => GPUDirectManager.EnableGPUDirect(backend));
        }

        [Fact]
        public void EnableGPUDirect_NullBackend_ThrowsArgumentNullException()
        {
            if (!GPUDirectManager.IsSupported)
            {
                // Skip test if GPU-direct is not supported
                return;
            }

            Assert.Throws<ArgumentNullException>(() => GPUDirectManager.EnableGPUDirect(null));
        }
    }
}
