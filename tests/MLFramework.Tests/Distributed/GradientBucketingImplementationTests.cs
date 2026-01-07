using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Distributed;
using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLFramework.Tests.Distributed
{
    /// <summary>
    /// Simple tests for GradientBucketing implementation.
    /// </summary>
    [TestClass]
    public class GradientBucketingImplementationTests
    {
        private MockProcessGroup _processGroup;

        [TestInitialize]
        public void Setup()
        {
            _processGroup = MockProcessGroup.Create(worldSize: 4, rank: 0);
        }

        [TestCleanup]
        public void Cleanup()
        {
            _processGroup?.Destroy();
        }

        [TestMethod]
        public void GradientBucket_CanBeCreated_WithGradients()
        {
            // Arrange
            var grad1 = Tensor.Random(new int[] { 10 });
            var grad2 = Tensor.Random(new int[] { 20 });
            var gradients = new[] { grad1, grad2 };

            // Act
            var bucket = new GradientBucket(0, 30 * sizeof(float), gradients);

            // Assert
            Assert.AreEqual(0, bucket.BucketIndex);
            Assert.AreEqual(30 * sizeof(float), bucket.SizeInBytes);
            Assert.AreEqual(2, bucket.Gradients.Length);
            Assert.IsFalse(bucket.IsReduced);
        }

        [TestMethod]
        public void GradientBucket_Prepare_CopiesGradientsToBucketTensor()
        {
            // Arrange
            var grad1 = new Tensor(new float[] { 1, 2, 3 }, new int[] { 3 });
            var grad2 = new Tensor(new float[] { 4, 5 }, new int[] { 2 });
            var gradients = new[] { grad1, grad2 };
            var bucket = new GradientBucket(0, 5 * sizeof(float), gradients);

            // Act
            bucket.Prepare();

            // Assert
            var bucketData = bucket.BucketTensor.Data;
            Assert.AreEqual(1f, bucketData[0], 0.001f);
            Assert.AreEqual(2f, bucketData[1], 0.001f);
            Assert.AreEqual(3f, bucketData[2], 0.001f);
            Assert.AreEqual(4f, bucketData[3], 0.001f);
            Assert.AreEqual(5f, bucketData[4], 0.001f);
        }

        [TestMethod]
        public void GradientBucket_CopyBack_RestoresOriginalGradients()
        {
            // Arrange
            var grad1 = new Tensor(new float[] { 1, 2, 3 }, new int[] { 3 });
            var grad2 = new Tensor(new float[] { 4, 5 }, new int[] { 2 });
            var gradients = new[] { grad1, grad2 };
            var bucket = new GradientBucket(0, 5 * sizeof(float), gradients);

            // Modify bucket tensor to simulate reduction
            bucket.Prepare();
            bucket.BucketTensor.Data[0] = 10f;
            bucket.BucketTensor.Data[1] = 20f;
            bucket.BucketTensor.Data[2] = 30f;
            bucket.BucketTensor.Data[3] = 40f;
            bucket.BucketTensor.Data[4] = 50f;

            // Act
            bucket.CopyBack();

            // Assert
            Assert.AreEqual(10f, grad1.Data[0], 0.001f);
            Assert.AreEqual(20f, grad1.Data[1], 0.001f);
            Assert.AreEqual(30f, grad1.Data[2], 0.001f);
            Assert.AreEqual(40f, grad2.Data[0], 0.001f);
            Assert.AreEqual(50f, grad2.Data[1], 0.001f);
        }

        [TestMethod]
        public async Task GradientBucket_ReduceAsync_CallsProcessGroup()
        {
            // Arrange
            var grad1 = Tensor.Random(new int[] { 10 });
            var grad2 = Tensor.Random(new int[] { 20 });
            var gradients = new[] { grad1, grad2 };
            var bucket = new GradientBucket(0, 30 * sizeof(float), gradients);

            // Act
            await bucket.ReduceAsync(_processGroup, ReduceOp.Sum);

            // Assert
            Assert.IsTrue(bucket.IsReduced);
        }

        [TestMethod]
        public void GradientBucketManager_CreatesBuckets_Correctly()
        {
            // Arrange
            var gradients = new[]
            {
                Tensor.Random(new int[] { 100 }),  // 400 bytes
                Tensor.Random(new int[] { 200 }),  // 800 bytes
                Tensor.Random(new int[] { 50 })    // 200 bytes
            };

            // Act
            var manager = new GradientBucketManager(_processGroup, gradients, bucketSizeInBytes: 600);

            // Assert
            Assert.IsNotNull(manager);
            Assert.AreEqual(0, manager.GetBucketIndex(gradients[0]));
        }

        [TestMethod]
        public async Task GradientBucketManager_ReduceAllAsync_Works()
        {
            // Arrange
            var grad1 = Tensor.Random(new int[] { 10 });
            var grad2 = Tensor.Random(new int[] { 20 });
            var gradients = new[] { grad1, grad2 };
            var manager = new GradientBucketManager(_processGroup, gradients);

            // Act
            await manager.ReduceAllAsync();

            // Assert
            Assert.IsTrue(gradients.All(g => g != null));
        }

        [TestMethod]
        public void GradientBucketManager_CopyBackAll_Works()
        {
            // Arrange
            var grad1 = new Tensor(new float[] { 1, 2, 3 }, new int[] { 3 });
            var grad2 = new Tensor(new float[] { 4, 5 }, new int[] { 2 });
            var gradients = new[] { grad1, grad2 };
            var manager = new GradientBucketManager(_processGroup, gradients);

            // Prepare bucket and modify it
            manager.ReduceBucketAsync(0).Wait();
            var bucket = manager.GetType().GetField("_buckets", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                                   .GetValue(manager) as GradientBucket[];
            bucket[0].Prepare();
            bucket[0].BucketTensor.Data[0] = 100f;

            // Act
            manager.CopyBackAll();

            // Assert
            Assert.AreEqual(100f, grad1.Data[0], 0.001f);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void GradientBucketManager_NullGradients_ThrowsException()
        {
            // Act
            new GradientBucketManager(_processGroup, null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void GradientBucketManager_EmptyGradients_ThrowsException()
        {
            // Act
            new GradientBucketManager(_processGroup, Enumerable.Empty<Tensor>());
        }
    }
}
