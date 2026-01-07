using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Distributed;
using MLFramework.Distributed.FSDP;
using Moq;

namespace MLFramework.Tests.Distributed.FSDP
{
    /// <summary>
    /// Integration tests for FSDP.
    /// </summary>
    [TestClass]
    public class FSDPIntegrationTests
    {
        [TestMethod]
        public void TestFSDPWrapperCreation()
        {
            var mockModel = new Mock<IModel>();
            var mockProcessGroup = new Mock<IProcessGroup>();

            mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            mockProcessGroup.Setup(p => p.Rank).Returns(0);
            mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            Assert.IsNotNull(fsdp);
            Assert.AreEqual(mockModel.Object, fsdp.Model);
            Assert.AreEqual(config, fsdp.Config);
            Assert.AreEqual(mockProcessGroup.Object, fsdp.ProcessGroup);
        }

        [TestMethod]
        public void TestFSDPWithSingleDevice()
        {
            var mockModel = new Mock<IModel>();
            var mockProcessGroup = new Mock<IProcessGroup>();

            mockProcessGroup.Setup(p => p.WorldSize).Returns(1);
            mockProcessGroup.Setup(p => p.Rank).Returns(0);
            mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            // Should not throw for single device
            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);
            Assert.IsNotNull(fsdp);
        }

        [TestMethod]
        public void TestFSDPWithMultipleDevices()
        {
            var mockModel = new Mock<IModel>();
            var mockProcessGroup = new Mock<IProcessGroup>();

            mockProcessGroup.Setup(p => p.WorldSize).Returns(8);
            mockProcessGroup.Setup(p => p.Rank).Returns(2);
            mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.LayerWise };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            Assert.IsNotNull(fsdp);
            Assert.AreEqual(8, fsdp.ProcessGroup.WorldSize);
            Assert.AreEqual(2, fsdp.ProcessGroup.Rank);
        }

        [TestMethod]
        public void TestFSDPWithDifferentShardingStrategies()
        {
            var mockModel = new Mock<IModel>();
            var mockProcessGroup = new Mock<IProcessGroup>();

            mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            mockProcessGroup.Setup(p => p.Rank).Returns(0);
            mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

            // Test Full sharding
            var configFull = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };
            var fsdpFull = new FSDP(mockModel.Object, configFull, mockProcessGroup.Object);
            Assert.IsNotNull(fsdpFull);

            // Test LayerWise sharding
            var configLayerWise = new FSDPConfig { ShardingStrategy = ShardingStrategy.LayerWise };
            var fsdpLayerWise = new FSDP(mockModel.Object, configLayerWise, mockProcessGroup.Object);
            Assert.IsNotNull(fsdpLayerWise);

            // Test Hybrid sharding
            var configHybrid = new FSDPConfig { ShardingStrategy = ShardingStrategy.Hybrid };
            var fsdpHybrid = new FSDP(mockModel.Object, configHybrid, mockProcessGroup.Object);
            Assert.IsNotNull(fsdpHybrid);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TestFSDPWithNullModel()
        {
            var mockProcessGroup = new Mock<IProcessGroup>();
            mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            mockProcessGroup.Setup(p => p.Rank).Returns(0);
            mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(null, config, mockProcessGroup.Object);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TestFSDPWithNullConfig()
        {
            var mockModel = new Mock<IModel>();
            var mockProcessGroup = new Mock<IProcessGroup>();
            mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            mockProcessGroup.Setup(p => p.Rank).Returns(0);
            mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

            var fsdp = new FSDP(mockModel.Object, null, mockProcessGroup.Object);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TestFSDPWithNullProcessGroup()
        {
            var mockModel = new Mock<IModel>();
            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config, null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TestFSDPWithInvalidConfig()
        {
            var mockModel = new Mock<IModel>();
            var mockProcessGroup = new Mock<IProcessGroup>();

            mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            mockProcessGroup.Setup(p => p.Rank).Returns(0);
            mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

            var config = new FSDPConfig { BucketSizeMB = 0 }; // Invalid

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);
        }

        [TestMethod]
        public void TestFSDPGetShardingUnits()
        {
            var mockModel = new Mock<IModel>();
            var mockProcessGroup = new Mock<IProcessGroup>();

            mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            mockProcessGroup.Setup(p => p.Rank).Returns(0);
            mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            var shardingUnits = fsdp.GetShardingUnits();

            Assert.IsNotNull(shardingUnits);
            // Initially empty, will be populated when sharding is implemented
            Assert.AreEqual(0, shardingUnits.Count);
        }

        [TestMethod]
        public void TestFSDPDispose()
        {
            var mockModel = new Mock<IModel>();
            var mockProcessGroup = new Mock<IProcessGroup>();

            mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            mockProcessGroup.Setup(p => p.Rank).Returns(0);
            mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            // Should not throw
            fsdp.Dispose();
        }

        [TestMethod]
        public void TestFSDPDisposeMultipleTimes()
        {
            var mockModel = new Mock<IModel>();
            var mockProcessGroup = new Mock<IProcessGroup>();

            mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            mockProcessGroup.Setup(p => p.Rank).Returns(0);
            mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            // Should not throw
            fsdp.Dispose();
            fsdp.Dispose();
        }

        [TestMethod]
        public void TestFSDPWithMixedPrecisionEnabled()
        {
            var mockModel = new Mock<IModel>();
            var mockProcessGroup = new Mock<IProcessGroup>();

            mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            mockProcessGroup.Setup(p => p.Rank).Returns(0);
            mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

            var config = new FSDPConfig
            {
                ShardingStrategy = ShardingStrategy.Full,
                MixedPrecision = true
            };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            Assert.IsNotNull(fsdp);
            Assert.IsTrue(fsdp.Config.MixedPrecision);
        }

        [TestMethod]
        public void TestFSDPWithCPUOffloadingEnabled()
        {
            var mockModel = new Mock<IModel>();
            var mockProcessGroup = new Mock<IProcessGroup>();

            mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            mockProcessGroup.Setup(p => p.Rank).Returns(0);
            mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

            var config = new FSDPConfig
            {
                ShardingStrategy = ShardingStrategy.Full,
                OffloadToCPU = true
            };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            Assert.IsNotNull(fsdp);
            Assert.IsTrue(fsdp.Config.OffloadToCPU);
        }

        [TestMethod]
        public void TestFSDPWithActivationCheckpointingEnabled()
        {
            var mockModel = new Mock<IModel>();
            var mockProcessGroup = new Mock<IProcessGroup>();

            mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            mockProcessGroup.Setup(p => p.Rank).Returns(0);
            mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

            var config = new FSDPConfig
            {
                ShardingStrategy = ShardingStrategy.Full,
                ActivationCheckpointing = true
            };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            Assert.IsNotNull(fsdp);
            Assert.IsTrue(fsdp.Config.ActivationCheckpointing);
        }

        [TestMethod]
        public void TestFSDPWithFullConfiguration()
        {
            var mockModel = new Mock<IModel>();
            var mockProcessGroup = new Mock<IProcessGroup>();

            mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            mockProcessGroup.Setup(p => p.Rank).Returns(0);
            mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

            var config = new FSDPConfig
            {
                ShardingStrategy = ShardingStrategy.Hybrid,
                MixedPrecision = true,
                OffloadToCPU = true,
                ActivationCheckpointing = true,
                BucketSizeMB = 50,
                NumCommunicationWorkers = 4
            };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            Assert.IsNotNull(fsdp);
            Assert.AreEqual(ShardingStrategy.Hybrid, fsdp.Config.ShardingStrategy);
            Assert.IsTrue(fsdp.Config.MixedPrecision);
            Assert.IsTrue(fsdp.Config.OffloadToCPU);
            Assert.IsTrue(fsdp.Config.ActivationCheckpointing);
            Assert.AreEqual(50, fsdp.Config.BucketSizeMB);
            Assert.AreEqual(4, fsdp.Config.NumCommunicationWorkers);
        }

        [TestMethod]
        public void TestFSDPModelProperty()
        {
            var mockModel = new Mock<IModel>();
            mockModel.Setup(m => m.Name).Returns("TestModel");

            var mockProcessGroup = new Mock<IProcessGroup>();
            mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            mockProcessGroup.Setup(p => p.Rank).Returns(0);
            mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            Assert.AreEqual(mockModel.Object, fsdp.Model);
            Assert.AreEqual("TestModel", fsdp.Model.Name);
        }
    }
}
