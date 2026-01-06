using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace MLFramework.Tests.Distributed
{
    [TestClass]
    public class ProcessGroupTests
    {
        [TestMethod]
        public void ProcessGroup_Create_WithValidParameters_ReturnsGroup()
        {
            var group = MockProcessGroup.Create(worldSize: 4, rank: 2);
            Assert.IsNotNull(group);
            group.Destroy();
        }

        [TestMethod]
        public void ProcessGroup_Rank_IsCorrect()
        {
            var group = MockProcessGroup.Create(worldSize: 4, rank: 2);
            Assert.AreEqual(2, group.Rank);
            group.Destroy();
        }

        [TestMethod]
        public void ProcessGroup_WorldSize_IsCorrect()
        {
            var group = MockProcessGroup.Create(worldSize: 8, rank: 0);
            Assert.AreEqual(8, group.WorldSize);
            group.Destroy();
        }

        [TestMethod]
        public void ProcessGroup_Backend_ReturnsValidBackend()
        {
            var group = MockProcessGroup.Create(worldSize: 2, rank: 0);
            Assert.IsNotNull(group.Backend);
            group.Destroy();
        }

        [TestMethod]
        public void ProcessGroup_Singleton_PreventsMultipleCreation()
        {
            var group1 = MockProcessGroup.Create(worldSize: 2, rank: 0);

            Assert.ThrowsException<InvalidOperationException>(() =>
            {
                var group2 = MockProcessGroup.Create(worldSize: 2, rank: 1);
            });

            group1.Destroy();
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

        [TestMethod]
        public void ProcessGroup_SingleRank_WorldSizeIsOne()
        {
            var group = MockProcessGroup.Create(worldSize: 1, rank: 0);
            Assert.AreEqual(1, group.WorldSize);
            Assert.AreEqual(0, group.Rank);
            group.Destroy();
        }

        [TestMethod]
        public void ProcessGroup_RankZero_IsRoot()
        {
            var group = MockProcessGroup.Create(worldSize: 4, rank: 0);
            Assert.AreEqual(0, group.Rank);
            group.Destroy();
        }

        [TestMethod]
        public void ProcessGroup_LastRank_IsWorldSizeMinusOne()
        {
            var group = MockProcessGroup.Create(worldSize: 8, rank: 7);
            Assert.AreEqual(7, group.Rank);
            Assert.AreEqual(8, group.WorldSize);
            group.Destroy();
        }
    }
}
