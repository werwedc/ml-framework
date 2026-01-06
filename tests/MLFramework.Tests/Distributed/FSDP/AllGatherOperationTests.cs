using Microsoft.VisualStudio.TestTools.UnitTesting;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using MLFramework.Distributed;
using MLFramework.Distributed.FSDP;
using Moq;

namespace MLFramework.Tests.Distributed.FSDP
{
    /// <summary>
    /// Unit tests for AllGatherOperation.
    /// </summary>
    [TestClass]
    public class AllGatherOperationTests
    {
        private Mock<IProcessGroup> _mockProcessGroup;

        [TestInitialize]
        public void Setup()
        {
            _mockProcessGroup = new Mock<IProcessGroup>();
        }

        [TestMethod]
        public void TestSingleDeviceAllGather()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(1);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var shard = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });
            var op = new AllGatherOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float32, 0);

            var result = op.AllGather(shard);

            Assert.AreEqual(3, result.Size);
            CollectionAssert.AreEqual(new[] { 1.0f, 2.0f, 3.0f }, result.Data);
        }

        [TestMethod]
        public void TestAllGatherMultipleDevices()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var shard = Tensor.FromArray(new[] { 1.0f, 2.0f });
            var op = new AllGatherOperation(_mockProcessGroup.Object, new[] { 8L }, DataType.Float32, 0);

            // Note: This test will need to mock the Send/Recv operations
            // For now, we just test the object creation and basic validation
            Assert.IsNotNull(op);
            Assert.AreEqual(8, op.GetGatheredBuffer().Size);
        }

        [TestMethod]
        public void TestAllGatherUnevenShards()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(3);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var shard = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var op = new AllGatherOperation(_mockProcessGroup.Object, new[] { 10L }, DataType.Float32, 0);

            // 10 elements / 3 devices = 4, 3, 3 elements per device
            Assert.IsNotNull(op);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TestNullTensor()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var op = new AllGatherOperation(_mockProcessGroup.Object, new[] { 8L }, DataType.Float32, 0);
            op.AllGather(null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TestNullProcessGroup()
        {
            var op = new AllGatherOperation(null, new[] { 8L }, DataType.Float32, 0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TestNullShape()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);

            var op = new AllGatherOperation(_mockProcessGroup.Object, null, DataType.Float32, 0);
        }

        [TestMethod]
        public void TestGetGatheredBufferReturnsCorrectShape()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var shape = new[] { 10L, 20L, 30L };
            var op = new AllGatherOperation(_mockProcessGroup.Object, shape, DataType.Float32, 0);

            var buffer = op.GetGatheredBuffer();

            Assert.IsNotNull(buffer);
            Assert.AreEqual(10, buffer.Shape[0]);
            Assert.AreEqual(20, buffer.Shape[1]);
            Assert.AreEqual(30, buffer.Shape[2]);
        }

        [TestMethod]
        public void TestAllGatherDifferentDataTypes()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(1);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            // Float32
            var floatShard = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });
            var floatOp = new AllGatherOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float32, 0);
            var floatResult = floatOp.AllGather(floatShard);
            Assert.AreEqual(DataType.Float32, floatResult.Dtype);

            // Float16 (if supported)
            var halfShard = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });
            var halfOp = new AllGatherOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float16, 0);
            var halfResult = halfOp.AllGather(halfShard);
            Assert.AreEqual(DataType.Float16, halfResult.Dtype);
        }

        [TestMethod]
        public void TestAllGatherMultipleBuckets()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(1);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var shard = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });

            // Create operations for different buckets
            var op0 = new AllGatherOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float32, 0);
            var op1 = new AllGatherOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float32, 1);
            var op2 = new AllGatherOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float32, 2);

            var result0 = op0.AllGather(shard);
            var result1 = op1.AllGather(shard);
            var result2 = op2.AllGather(shard);

            // All should produce valid results
            Assert.IsNotNull(result0);
            Assert.IsNotNull(result1);
            Assert.IsNotNull(result2);
        }

        [TestMethod]
        public void TestAllGatherLargeTensor()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(1);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var largeData = new float[10000];
            for (int i = 0; i < largeData.Length; i++)
            {
                largeData[i] = i;
            }

            var shard = Tensor.FromArray(largeData);
            var op = new AllGatherOperation(_mockProcessGroup.Object, new[] { 10000L }, DataType.Float32, 0);

            var result = op.AllGather(shard);

            Assert.AreEqual(10000, result.Size);
            CollectionAssert.AreEqual(largeData, result.Data);
        }
    }
}
