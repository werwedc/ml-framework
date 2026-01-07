using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Distributed.FSDP;
using Moq;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Distributed.FSDP
{
    /// <summary>
    /// Tests for the FSDP wrapper implementation.
    /// </summary>
    [TestClass]
    public class FSDPWrapperTests
    {
        private Mock<IProcessGroup> CreateMockProcessGroup(int worldSize = 4, int rank = 0)
        {
            var mock = new Mock<IProcessGroup>();
            mock.Setup(p => p.WorldSize).Returns(worldSize);
            mock.Setup(p => p.Rank).Returns(rank);
            mock.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());
            return mock;
        }

        private Mock<IModel> CreateMockModel(string name = "TestModel", params NamedTensor[] parameters)
        {
            var mock = new Mock<IModel>();
            mock.Setup(m => m.Name).Returns(name);
            mock.Setup(m => m.GetParameters()).Returns(new List<NamedTensor>(parameters));
            mock.Setup(m => m.Forward(It.IsAny<Tensor>())).Returns(new Tensor(new float[] { 1f }, new[] { 1 }));
            return mock;
        }

        [TestMethod]
        public void TestFSDPWrapperCreation()
        {
            var mockProcessGroup = CreateMockProcessGroup();
            var mockModel = CreateMockModel();
            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            Assert.IsNotNull(fsdp);
            Assert.AreEqual("TestModel", fsdp.Name);
            Assert.AreEqual(mockModel.Object, fsdp.Model);
            Assert.AreEqual(config, fsdp.Config);
            Assert.AreEqual(mockProcessGroup.Object, fsdp.ProcessGroup);
        }

        [TestMethod]
        public void TestFSDPWithParameters()
        {
            var mockProcessGroup = CreateMockProcessGroup();
            var param1 = new NamedTensor("layer1.weight", new Tensor(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 }));
            var param2 = new NamedTensor("layer1.bias", new Tensor(new float[] { 1f }, new[] { 1 }));
            var mockModel = CreateMockModel("TestModel", param1, param2);
            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            Assert.IsNotNull(fsdp);
            var parameters = fsdp.GetParameters();
            Assert.IsNotNull(parameters);
            // Parameters should be sharded, so we should have sharding units
            var shardingUnits = fsdp.GetShardingUnits();
            Assert.IsNotNull(shardingUnits);
            // Should have 2 sharding units (one for each parameter)
            Assert.AreEqual(2, shardingUnits.Count);
        }

        [TestMethod]
        public void TestFSDPForwardPass()
        {
            var mockProcessGroup = CreateMockProcessGroup();
            var mockModel = CreateMockModel();
            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);
            var input = new Tensor(new float[] { 1f, 2f }, new[] { 2 });

            var output = fsdp.Forward(input);

            Assert.IsNotNull(output);
            mockModel.Verify(m => m.Forward(It.IsAny<Tensor>()), Times.Once);
        }

        [TestMethod]
        public void TestFSDPBackwardPass()
        {
            var mockProcessGroup = CreateMockProcessGroup();
            var mockModel = CreateMockModel();
            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            // Should not throw
            fsdp.Backward();

            mockModel.Verify(m => m.Backward(), Times.Once);
        }

        [TestMethod]
        public void TestFSDPGetGradients()
        {
            var mockProcessGroup = CreateMockProcessGroup();
            var param1 = new NamedTensor("layer1.weight", new Tensor(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 }));
            var mockModel = CreateMockModel("TestModel", param1);
            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            var gradients = fsdp.GetGradients();

            Assert.IsNotNull(gradients);
            // Initially empty until gradients are computed
            Assert.AreEqual(0, gradients.Count);
        }

        [TestMethod]
        public void TestFSDPDispose()
        {
            var mockProcessGroup = CreateMockProcessGroup();
            var mockModel = CreateMockModel();
            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            // Should not throw
            fsdp.Dispose();
        }

        [TestMethod]
        [ExpectedException(typeof(ObjectDisposedException))]
        public void TestFSDPForwardAfterDispose()
        {
            var mockProcessGroup = CreateMockProcessGroup();
            var mockModel = CreateMockModel();
            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);
            var input = new Tensor(new float[] { 1f }, new[] { 1 });

            fsdp.Dispose();

            // Should throw ObjectDisposedException
            fsdp.Forward(input);
        }

        [TestMethod]
        [ExpectedException(typeof(ObjectDisposedException))]
        public void TestFSDPBackwardAfterDispose()
        {
            var mockProcessGroup = CreateMockProcessGroup();
            var mockModel = CreateMockModel();
            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            fsdp.Dispose();

            // Should throw ObjectDisposedException
            fsdp.Backward();
        }

        [TestMethod]
        [ExpectedException(typeof(ObjectDisposedException))]
        public void TestFSDPGetParametersAfterDispose()
        {
            var mockProcessGroup = CreateMockProcessGroup();
            var mockModel = CreateMockModel();
            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            fsdp.Dispose();

            // Should throw ObjectDisposedException
            fsdp.GetParameters();
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TestFSDPWithNullModel()
        {
            var mockProcessGroup = CreateMockProcessGroup();
            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(null, config, mockProcessGroup.Object);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TestFSDPWithNullConfig()
        {
            var mockProcessGroup = CreateMockProcessGroup();
            var mockModel = CreateMockModel();

            var fsdp = new FSDP(mockModel.Object, null, mockProcessGroup.Object);
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void TestFSDPWithNoDefaultProcessGroup()
        {
            // Ensure default process group is null
            MLFramework.Distributed.ProcessGroup.Destroy();

            var mockModel = CreateMockModel();
            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config);
        }

        [TestMethod]
        public void TestFSDPWithDifferentShardingStrategies()
        {
            var mockProcessGroup = CreateMockProcessGroup();
            var mockModel = CreateMockModel();

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
        public void TestFSDPWithAlwaysGatheredParameters()
        {
            var mockProcessGroup = CreateMockProcessGroup();
            // Embedding should always be gathered
            var param1 = new NamedTensor("embedding.weight", new Tensor(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 }));
            var param2 = new NamedTensor("layer1.weight", new Tensor(new float[] { 1f, 2f }, new[] { 2 }));
            var mockModel = CreateMockModel("TestModel", param1, param2);
            var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

            var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

            var shardingUnits = fsdp.GetShardingUnits();
            // Should only have 1 sharding unit (layer1.weight)
            // embedding.weight should be in AlwaysGathered list
            Assert.AreEqual(1, shardingUnits.Count);
        }
    }
}
