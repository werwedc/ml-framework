using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace MLFramework.Tests.Distributed
{
    /// <summary>
    /// Mock Gloo backend for testing.
    /// </summary>
    public class MockGlooBackend : ICommunicationBackend
    {
        public string Name => "Gloo";
        public bool IsAvailable => true; // Available in mock environment
        public int DeviceCount => 1;
        public bool SupportsAsync => true;
        public bool SupportsGPUDirect => false;
        public long GetBufferSizeLimit() => 512 * 1024 * 1024; // 512MB
    }

    [TestClass]
    public class GlooBackendTests
    {
        [TestMethod]
        public void GlooBackend_Name_IsCorrect()
        {
            var backend = new MockGlooBackend();
            Assert.AreEqual("Gloo", backend.Name);
        }

        [TestMethod]
        public void GlooBackend_IsAvailable_ReturnsTrue()
        {
            var backend = new MockGlooBackend();
            Assert.IsTrue(backend.IsAvailable, "Gloo should be available in mock environment");
        }

        [TestMethod]
        public void GlooBackend_SupportsAsync_ReturnsTrue()
        {
            var backend = new MockGlooBackend();
            Assert.IsTrue(backend.SupportsAsync);
        }

        [TestMethod]
        public void GlooBackend_SupportsGPUDirect_ReturnsFalse()
        {
            var backend = new MockGlooBackend();
            Assert.IsFalse(backend.SupportsGPUDirect, "Gloo does not support GPU Direct");
        }

        [TestMethod]
        public void GlooBackend_DeviceCount_ReturnsOne()
        {
            var backend = new MockGlooBackend();
            Assert.AreEqual(1, backend.DeviceCount);
        }

        [TestMethod]
        public void GlooBackend_GetBufferSizeLimit_ReturnsValidLimit()
        {
            var backend = new MockGlooBackend();
            var limit = backend.GetBufferSizeLimit();
            Assert.IsTrue(limit > 0, "Buffer limit should be positive");
            Assert.AreEqual(512 * 1024 * 1024, limit, "Buffer limit should be 512MB");
        }

        [TestMethod]
        public void GlooBackend_LowerBufferSizeThanNCCL()
        {
            var ncclBackend = new MockNCCLBackend();
            var glooBackend = new MockGlooBackend();

            Assert.IsTrue(
                glooBackend.GetBufferSizeLimit() < ncclBackend.GetBufferSizeLimit(),
                "Gloo buffer limit should be lower than NCCL"
            );
        }
    }
}
