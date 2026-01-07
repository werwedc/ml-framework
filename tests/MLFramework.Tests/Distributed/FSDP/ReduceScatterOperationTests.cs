using Microsoft.VisualStudio.TestTools.UnitTesting;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using MLFramework.Distributed;
using MLFramework.Distributed.FSDP;
using Moq;

namespace MLFramework.Tests.Distributed.FSDP
{
    /// <summary>
    /// Unit tests for ReduceScatterOperation.
    /// </summary>
    [TestClass]
    public class ReduceScatterOperationTests
    {
        private Mock<IProcessGroup> _mockProcessGroup;

        [TestInitialize]
        public void Setup()
        {
            _mockProcessGroup = new Mock<IProcessGroup>();
        }

        [TestMethod]
        public void TestSingleDeviceReduceScatter()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(1);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var fullTensor = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });
            var op = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float32, 0);

            var result = op.ReduceScatter(fullTensor);

            Assert.AreEqual(3, result.Size);
            CollectionAssert.AreEqual(new[] { 1.0f, 2.0f, 3.0f }, result.Data);
        }

        [TestMethod]
        public void TestReduceScatterSum()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var fullTensor = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var op = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 4L }, DataType.Float32, 0);

            // Test basic validation
            Assert.IsNotNull(op);
            Assert.AreEqual(ReduceOp.Sum, op.GetReduceOp());
        }

        [TestMethod]
        public void TestReduceScatterAvg()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var fullTensor = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var op = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 4L }, DataType.Float32, 0, ReduceOp.Avg);

            Assert.AreEqual(ReduceOp.Avg, op.GetReduceOp());
        }

        [TestMethod]
        public void TestReduceScatterMax()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var fullTensor = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var op = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 4L }, DataType.Float32, 0, ReduceOp.Max);

            Assert.AreEqual(ReduceOp.Max, op.GetReduceOp());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TestNullTensor()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var op = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 4L }, DataType.Float32, 0);
            op.ReduceScatter(null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TestNullProcessGroup()
        {
            var op = new ReduceScatterOperation(null, new[] { 4L }, DataType.Float32, 0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TestNullShape()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);

            var op = new ReduceScatterOperation(_mockProcessGroup.Object, null, DataType.Float32, 0);
        }

        [TestMethod]
        public void TestReduceScatterMin()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var fullTensor = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var op = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 4L }, DataType.Float32, 0, ReduceOp.Min);

            Assert.AreEqual(ReduceOp.Min, op.GetReduceOp());
        }

        [TestMethod]
        public void TestReduceScatterProduct()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var fullTensor = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var op = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 4L }, DataType.Float32, 0, ReduceOp.Product);

            Assert.AreEqual(ReduceOp.Product, op.GetReduceOp());
        }

        [TestMethod]
        public void TestReduceScatterDifferentRanks()
        {
            // Test rank 0
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(2);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var fullTensor = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var op0 = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 4L }, DataType.Float32, 0);
            Assert.AreEqual(ReduceOp.Sum, op0.GetReduceOp());

            // Test rank 1
            _mockProcessGroup.Setup(p => p.Rank).Returns(1);
            var op1 = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 4L }, DataType.Float32, 0);
            Assert.AreEqual(ReduceOp.Sum, op1.GetReduceOp());
        }

        [TestMethod]
        public void TestReduceScatterMultipleBuckets()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(1);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var fullTensor = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });

            // Create operations for different buckets
            var op0 = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float32, 0);
            var op1 = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float32, 1);
            var op2 = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float32, 2);

            var result0 = op0.ReduceScatter(fullTensor);
            var result1 = op1.ReduceScatter(fullTensor);
            var result2 = op2.ReduceScatter(fullTensor);

            // All should produce valid results
            Assert.IsNotNull(result0);
            Assert.IsNotNull(result1);
            Assert.IsNotNull(result2);
        }

        [TestMethod]
        public void TestReduceScatterLargeTensor()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(1);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var largeData = new float[10000];
            for (int i = 0; i < largeData.Length; i++)
            {
                largeData[i] = i;
            }

            var fullTensor = Tensor.FromArray(largeData);
            var op = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 10000L }, DataType.Float32, 0);

            var result = op.ReduceScatter(fullTensor);

            Assert.AreEqual(10000, result.Size);
            CollectionAssert.AreEqual(largeData, result.Data);
        }

        [TestMethod]
        public void TestReduceScatterDifferentDataTypes()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(1);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            // Float32
            var floatTensor = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });
            var floatOp = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float32, 0);
            var floatResult = floatOp.ReduceScatter(floatTensor);
            Assert.AreEqual(DataType.Float32, floatResult.Dtype);

            // Float16 (if supported)
            var halfTensor = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });
            var halfOp = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float16, 0);
            var halfResult = halfOp.ReduceScatter(halfTensor);
            Assert.AreEqual(DataType.Float16, halfResult.Dtype);
        }
    }
}
