using Microsoft.VisualStudio.TestTools.UnitTesting;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using MLFramework.Distributed;
using MLFramework.Distributed.FSDP;
using Moq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

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

            var shardData = new[] { 1.0f, 2.0f, 3.0f };
            var shard = new Tensor(shardData, new[] { 3 });
            var op = new AllGatherOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float32, 0);

            var result = op.AllGather(shard);

            Assert.AreEqual(3, result.Size);
            CollectionAssert.AreEqual(shardData, result.Data);
        }

        [TestMethod]
        public void TestAllGatherEqualSizedShards()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            // Each shard has 2 elements, total 8 elements
            var shardData = new[] { 1.0f, 2.0f };
            var shard = new Tensor(shardData, new[] { 2 });

            // Mock the Recv operations for the other 3 devices
            var recvCallCount = 0;
            _mockProcessGroup.Setup(p => p.Recv(It.IsAny<Tensor>(), It.IsAny<int>()))
                .Callback<Tensor, int>((tensor, srcRank) =>
                {
                    recvCallCount++;
                    // Simulate receiving data from other ranks
                    var data = tensor.Data;
                    for (int i = 0; i < data.Length; i++)
                    {
                        data[i] = (srcRank + 1) * 10.0f + i;
                    }
                });

            var op = new AllGatherOperation(_mockProcessGroup.Object, new[] { 8L }, DataType.Float32, 0);
            var result = op.AllGather(shard);

            // Verify that Recv was called 3 times (once for each other device)
            Assert.AreEqual(3, recvCallCount);
            Assert.AreEqual(8, result.Size);
            Assert.IsNotNull(result);
        }

        [TestMethod]
        public void TestAllGatherUnevenShards()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(3);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            // 10 elements / 3 devices = 4, 3, 3 elements per device (ceil division)
            var shardData = new[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var shard = new Tensor(shardData, new[] { 4 });

            var recvCallCount = 0;
            _mockProcessGroup.Setup(p => p.Recv(It.IsAny<Tensor>(), It.IsAny<int>()))
                .Callback<Tensor, int>((tensor, srcRank) =>
                {
                    recvCallCount++;
                    var data = tensor.Data;
                    for (int i = 0; i < data.Length; i++)
                    {
                        data[i] = (srcRank + 1) * 10.0f + i;
                    }
                });

            var op = new AllGatherOperation(_mockProcessGroup.Object, new[] { 10L }, DataType.Float32, 0);
            var result = op.AllGather(shard);

            // Verify that Recv was called 2 times (once for each other device)
            Assert.AreEqual(2, recvCallCount);
            Assert.AreEqual(10, result.Size);
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
        public void TestAllGatherDifferentDataTypes()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(1);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            // Float32
            var floatShardData = new[] { 1.0f, 2.0f, 3.0f };
            var floatShard = new Tensor(floatShardData, new[] { 3 });
            var floatOp = new AllGatherOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float32, 0);
            var floatResult = floatOp.AllGather(floatShard);
            Assert.AreEqual(DataType.Float32, floatResult.Dtype);

            // Float16 (if supported)
            var halfShardData = new[] { 1.0f, 2.0f, 3.0f };
            var halfShard = new Tensor(halfShardData, new[] { 3 });
            var halfOp = new AllGatherOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float16, 0);
            var halfResult = halfOp.AllGather(halfShard);
            Assert.AreEqual(DataType.Float16, halfResult.Dtype);
        }

        [TestMethod]
        public void TestAllGatherAsync()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(1);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var shardData = new[] { 1.0f, 2.0f, 3.0f };
            var shard = new Tensor(shardData, new[] { 3 });
            var op = new AllGatherOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float32, 0);

            var task = op.AllGatherAsync(shard);
            var result = task.Result;

            Assert.AreEqual(3, result.Size);
            CollectionAssert.AreEqual(shardData, result.Data);
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

            var shard = new Tensor(largeData, new[] { 10000 });
            var op = new AllGatherOperation(_mockProcessGroup.Object, new[] { 10000L }, DataType.Float32, 0);

            var result = op.AllGather(shard);

            Assert.AreEqual(10000, result.Size);
            CollectionAssert.AreEqual(largeData, result.Data);
        }

        [TestMethod]
        public void TestAllGatherOperationDispose()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(1);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var op = new AllGatherOperation(_mockProcessGroup.Object, new[] { 3L }, DataType.Float32, 0);

            // Should not throw
            op.Dispose();
        }
    }

    /// <summary>
    /// Unit tests for AllGatherHelper.
    /// </summary>
    [TestClass]
    public class AllGatherHelperTests
    {
        private Mock<IProcessGroup> _mockProcessGroup;

        [TestInitialize]
        public void Setup()
        {
            _mockProcessGroup = new Mock<IProcessGroup>();
        }

        [TestMethod]
        public async Task TestAllGatherMultipleAsync()
        {
            _mockProcessGroup.Setup(p => p.WorldSize).Returns(1);
            _mockProcessGroup.Setup(p => p.Rank).Returns(0);

            var tensor1 = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });
            var tensor2 = new Tensor(new[] { 4.0f, 5.0f, 6.0f }, new[] { 3 });
            var tensor3 = new Tensor(new[] { 7.0f, 8.0f, 9.0f }, new[] { 3 });

            var tensors = new List<Tensor> { tensor1, tensor2, tensor3 };

            var result = await AllGatherHelper.AllGatherMultipleAsync(_mockProcessGroup.Object, tensors);

            Assert.AreEqual(3, result.Count);
            Assert.AreEqual(3, result[0].Size);
            Assert.AreEqual(3, result[1].Size);
            Assert.AreEqual(3, result[2].Size);
        }

        [TestMethod]
        public async Task TestAllGatherMultipleAsyncEmptyList()
        {
            var result = await AllGatherHelper.AllGatherMultipleAsync(_mockProcessGroup.Object, new List<Tensor>());

            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Count);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public async Task TestAllGatherMultipleAsyncNullList()
        {
            await AllGatherHelper.AllGatherMultipleAsync(_mockProcessGroup.Object, null);
        }

        [TestMethod]
        public void TestCalculateBuckets()
        {
            var worldSize = 4;
            var parameterSizes = new List<long> { 1000, 2000, 3000, 4000, 5000 };
            var maxBucketSizeMB = 1; // 1 MB

            var buckets = AllGatherHelper.CalculateBuckets(worldSize, parameterSizes, maxBucketSizeMB);

            Assert.IsNotNull(buckets);
            Assert.AreEqual(5, buckets.Count);

            // All parameters should fit in bucket 0 (1000*4 + 2000*4 + ... < 1MB)
            Assert.AreEqual(0, buckets[0]);
            Assert.AreEqual(0, buckets[1]);
            Assert.AreEqual(0, buckets[2]);
            Assert.AreEqual(0, buckets[3]);
            Assert.AreEqual(0, buckets[4]);
        }

        [TestMethod]
        public void TestCalculateBucketsWithOverflow()
        {
            var worldSize = 2;
            var parameterSizes = new List<long> { 1000000, 1000000, 1000000 }; // 1M each
            var maxBucketSizeMB = 1; // 1 MB

            var buckets = AllGatherHelper.CalculateBuckets(worldSize, parameterSizes, maxBucketSizeMB);

            Assert.IsNotNull(buckets);
            Assert.AreEqual(3, buckets.Count);

            // Each parameter (gathered) is 2MB, so each goes in its own bucket
            Assert.AreEqual(0, buckets[0]);
            Assert.AreEqual(1, buckets[1]);
            Assert.AreEqual(2, buckets[2]);
        }

        [TestMethod]
        public void TestCalculateBucketsSingleDevice()
        {
            var worldSize = 1;
            var parameterSizes = new List<long> { 500000, 500000, 500000 }; // 500K each
            var maxBucketSizeMB = 1; // 1 MB

            var buckets = AllGatherHelper.CalculateBuckets(worldSize, parameterSizes, maxBucketSizeMB);

            Assert.IsNotNull(buckets);
            Assert.AreEqual(3, buckets.Count);

            // With world size 1, no gathering needed, so all fit in bucket 0
            Assert.AreEqual(0, buckets[0]);
            Assert.AreEqual(0, buckets[1]);
            Assert.AreEqual(0, buckets[2]);
        }

        [TestMethod]
        public void TestCalculateBucketsMixedSizes()
        {
            var worldSize = 4;
            var parameterSizes = new List<long> { 100000, 200000, 100000, 500000 }; // Different sizes
            var maxBucketSizeMB = 1; // 1 MB

            var buckets = AllGatherHelper.CalculateBuckets(worldSize, parameterSizes, maxBucketSizeMB);

            Assert.IsNotNull(buckets);
            Assert.AreEqual(4, buckets.Count);

            // Verify buckets are assigned correctly
            // After gathering: 400K, 800K, 400K, 2MB
            // With 1MB limit: 0, 0, 1, 2
            Assert.AreEqual(0, buckets[0]); // 400K fits
            Assert.AreEqual(0, buckets[1]); // 400K + 800K > 1MB? No, 400K + 800K = 1.2MB > 1MB, so new bucket
        }
    }
}
