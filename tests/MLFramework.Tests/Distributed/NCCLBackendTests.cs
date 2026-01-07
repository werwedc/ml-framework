using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace MLFramework.Tests.Distributed
{
    /// <summary>
    /// Mock NCCL backend for testing.
    /// </summary>
    public class MockNCCLBackend : ICommunicationBackend
    {
        public string Name => "NCCL";
        public bool IsAvailable => false; // Not available in mock environment
        public int DeviceCount => 0;
        public bool SupportsAsync => true;
        public bool SupportsGPUDirect => true;
        public long GetBufferSizeLimit() => 1024 * 1024 * 1024; // 1GB
    }

    [TestClass]
    public class NCCLBackendTests
    {
        [TestMethod]
        public void NCCLBackend_Name_IsCorrect()
        {
            var backend = new MockNCCLBackend();
            Assert.AreEqual("NCCL", backend.Name);
        }

        [TestMethod]
        public void NCCLBackend_NotAvailable_InMockEnvironment()
        {
            var backend = new MockNCCLBackend();
            Assert.IsFalse(backend.IsAvailable, "NCCL should not be available in mock environment");
        }

        [TestMethod]
        public void NCCLBackend_SupportsAsync_ReturnsTrue()
        {
            var backend = new MockNCCLBackend();
            Assert.IsTrue(backend.SupportsAsync);
        }

        [TestMethod]
        public void NCCLBackend_SupportsGPUDirect_ReturnsTrue()
        {
            var backend = new MockNCCLBackend();
            Assert.IsTrue(backend.SupportsGPUDirect);
        }

        [TestMethod]
        public void NCCLBackend_DeviceCount_ReturnsZero()
        {
            var backend = new MockNCCLBackend();
            Assert.AreEqual(0, backend.DeviceCount, "No devices available in mock environment");
        }

        [TestMethod]
        public void NCCLBackend_GetBufferSizeLimit_ReturnsValidLimit()
        {
            var backend = new MockNCCLBackend();
            var limit = backend.GetBufferSizeLimit();
            Assert.IsTrue(limit > 0, "Buffer limit should be positive");
            Assert.AreEqual(1024 * 1024 * 1024, limit, "Buffer limit should be 1GB");
        }
    }
}
