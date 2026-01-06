using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace MLFramework.Tests.Distributed
{
    [TestClass]
    public class CommunicationBackendTests
    {
        [TestMethod]
        public void MockBackend_Availability_ReturnsTrue()
        {
            var backend = new MockBackend();
            Assert.IsTrue(backend.IsAvailable);
        }

        [TestMethod]
        public void MockBackend_Name_IsCorrect()
        {
            var backend = new MockBackend();
            Assert.AreEqual("MockBackend", backend.Name);
        }

        [TestMethod]
        public void MockBackend_SupportsAsync_ReturnsTrue()
        {
            var backend = new MockBackend();
            Assert.IsTrue(backend.SupportsAsync);
        }

        [TestMethod]
        public void ProcessGroup_Singleton_OnlyOneActiveGroup()
        {
            // Should not be able to create multiple active process groups
            var group1 = MockProcessGroup.Create(worldSize: 2, rank: 0);

            Assert.ThrowsException<InvalidOperationException>(() =>
            {
                var group2 = MockProcessGroup.Create(worldSize: 2, rank: 1);
            });

            group1.Destroy();
        }

        [TestMethod]
        public void ProcessGroup_RankAndWorldSize_AreCorrect()
        {
            var group = MockProcessGroup.Create(worldSize: 4, rank: 2);
            Assert.AreEqual(2, group.Rank);
            Assert.AreEqual(4, group.WorldSize);
            group.Destroy();
        }

        [TestMethod]
        public void ProcessGroup_Backend_ReturnsMockBackend()
        {
            var group = MockProcessGroup.Create(worldSize: 2, rank: 0);
            Assert.IsInstanceOfType(group.Backend, typeof(MockBackend));
            group.Destroy();
        }

        [TestMethod]
        public void ProcessGroup_Destroy_AllowsNewCreation()
        {
            var group1 = MockProcessGroup.Create(worldSize: 2, rank: 0);
            group1.Destroy();

            var group2 = MockProcessGroup.Create(worldSize: 2, rank: 0);
            Assert.IsNotNull(group2);
            group2.Destroy();
        }

        [TestMethod]
        public void ProcessGroup_Barbar_CompletesSuccessfully()
        {
            var group = MockProcessGroup.Create(worldSize: 2, rank: 0);
            group.Barrier(); // Should not throw
            group.Destroy();
        }

        [TestMethod]
        public void ProcessGroup_BarbarAsync_CompletesSuccessfully()
        {
            var group = MockProcessGroup.Create(worldSize: 2, rank: 0);
            var task = group.BarrierAsync();
            Assert.IsTrue(task.IsCompleted);
            group.Destroy();
        }
    }
}
