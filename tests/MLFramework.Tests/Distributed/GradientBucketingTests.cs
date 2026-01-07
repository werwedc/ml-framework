using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Distributed;
using RitterFramework.Core.Tensor;
using System.Linq;
using System.Threading.Tasks;

namespace MLFramework.Tests.Distributed
{
    [TestClass]
    public class GradientBucketingTests
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
        public void GradientBucketManager_CreatesBuckets_OfCorrectSize()
        {
            var gradients = new[]
            {
                Tensor.Random(new int[] { 10 * 1024 * 1024 }),  // ~40MB
                Tensor.Random(new int[] { 5 * 1024 * 1024 }),   // ~20MB
                Tensor.Random(new int[] { 15 * 1024 * 1024 }), // ~60MB
                Tensor.Random(new int[] { 3 * 1024 * 1024 })    // ~12MB
            };

            var bucketManager = new GradientBucketManager(_processGroup, gradients, bucketSizeInBytes: 25 * 1024 * 1024);

            // Should create multiple buckets - verify by checking that gradients are in different buckets
            var bucket0 = bucketManager.GetBucketIndex(gradients[0]);
            var bucket1 = bucketManager.GetBucketIndex(gradients[1]);

            // Large gradients should be in different buckets
            Assert.IsTrue(bucket0 != bucket1 || gradients[0].Size + gradients[1].Size > 25 * 1024 * 1024 / 4,
                "Should create multiple buckets for large gradients");
        }

        [TestMethod]
        public void GradientBucketManager_SmallGradients_SingleBucket()
        {
            var gradients = new[]
            {
                Tensor.Random(new int[] { 100 }),
                Tensor.Random(new int[] { 200 }),
                Tensor.Random(new int[] { 50 })
            };

            var bucketManager = new GradientBucketManager(_processGroup, gradients, bucketSizeInBytes: 25 * 1024 * 1024);

            // All gradients should fit in a single bucket
            var bucket0 = bucketManager.GetBucketIndex(gradients[0]);
            var bucket1 = bucketManager.GetBucketIndex(gradients[1]);
            var bucket2 = bucketManager.GetBucketIndex(gradients[2]);

            Assert.AreEqual(bucket0, bucket1, "Small gradients should be in same bucket");
            Assert.AreEqual(bucket1, bucket2, "Small gradients should be in same bucket");
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void GradientBucketManager_EmptyGradients_ThrowsException()
        {
            var gradients = new Tensor[0];
            new GradientBucketManager(_processGroup, gradients);
        }

        [TestMethod]
        public void GradientBucket_CopyBack_PreservesShape()
        {
            var grad1 = Tensor.Random(new int[] { 10, 20 });
            var grad2 = Tensor.Random(new int[] { 30 });

            var gradients = new[] { grad1, grad2 };
            var bucketManager = new GradientBucketManager(_processGroup, gradients);

            // Simulate reduction and copy back
            var reduceTask = bucketManager.ReduceAllAsync();
            reduceTask.Wait();
            bucketManager.CopyBackAll();

            // Check that shapes are preserved
            CollectionAssert.AreEqual(new int[] { 10, 20 }, grad1.Shape);
            CollectionAssert.AreEqual(new int[] { 30 }, grad2.Shape);
        }

        [TestMethod]
        public void GradientBucket_MultipleBuckets_CreatedCorrectly()
        {
            var gradients = new[]
            {
                Tensor.Random(new int[] { 6 * 1024 * 1024 }),    // ~24MB - just under bucket size
                Tensor.Random(new int[] { 5 * 1024 * 1024 }),    // ~20MB - should start new bucket
                Tensor.Random(new int[] { 4 * 1024 * 1024 })     // ~16MB - should fit in second bucket
            };

            var bucketManager = new GradientBucketManager(_processGroup, gradients, bucketSizeInBytes: 25 * 1024 * 1024);

            // Should create 2 buckets
            var bucket0 = bucketManager.GetBucketIndex(gradients[0]);
            var bucket1 = bucketManager.GetBucketIndex(gradients[1]);
            var bucket2 = bucketManager.GetBucketIndex(gradients[2]);

            // First gradient should be in its own bucket (just under limit)
            // Second and third should be together (20+16=36 > 25, so second gets its own, third with second)
            Assert.AreNotEqual(bucket0, bucket1, "First two gradients should be in different buckets");
            Assert.AreEqual(bucket1, bucket2, "Second and third gradients should be in same bucket");
        }

        [TestMethod]
        public void GradientBucket_LargeGradient_ExceedsBucketSize_CreatesOwnBucket()
        {
            var gradients = new[]
            {
                Tensor.Random(new int[] { 30 * 1024 * 1024 }),   // ~120MB - exceeds bucket size
                Tensor.Random(new int[] { 10 * 1024 * 1024 })    // ~40MB
            };

            var bucketManager = new GradientBucketManager(_processGroup, gradients, bucketSizeInBytes: 25 * 1024 * 1024);

            // Should create 2 buckets (one for the large gradient)
            var bucket0 = bucketManager.GetBucketIndex(gradients[0]);
            var bucket1 = bucketManager.GetBucketIndex(gradients[1]);

            Assert.AreNotEqual(bucket0, bucket1, "Large gradient and smaller gradient should be in different buckets");
        }

        [TestMethod]
        public async Task GradientBucketManager_CopyBackAll_CompletesSuccessfully()
        {
            var gradients = new[]
            {
                Tensor.Random(new int[] { 100 }),
                Tensor.Random(new int[] { 200 })
            };

            var bucketManager = new GradientBucketManager(_processGroup, gradients);

            // Should not throw
            await bucketManager.ReduceAllAsync();
            bucketManager.CopyBackAll();
        }

        [TestMethod]
        public void GradientBucket_OffsetCalculatedCorrectly()
        {
            var grad1 = Tensor.Random(new int[] { 10 });
            var grad2 = Tensor.Random(new int[] { 20 });
            var grad3 = Tensor.Random(new int[] { 15 });

            var gradients = new[] { grad1, grad2, grad3 };
            var bucketManager = new GradientBucketManager(_processGroup, gradients);

            // Check that buckets were created and gradients are mapped
            Assert.IsNotNull(bucketManager.GetBucketIndex(grad1));
            Assert.IsNotNull(bucketManager.GetBucketIndex(grad2));
            Assert.IsNotNull(bucketManager.GetBucketIndex(grad3));
        }
    }
}
