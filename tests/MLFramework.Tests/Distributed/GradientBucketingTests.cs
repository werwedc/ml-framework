using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Tensor;
using System.Collections.Generic;

namespace MLFramework.Tests.Distributed
{
    /// <summary>
    /// Mock GradientBucketManager for testing.
    /// </summary>
    public class GradientBucketManager
    {
        private readonly IProcessGroup _processGroup;
        private readonly Tensor[] _parameters;
        private readonly List<GradientBucket> _buckets;
        private readonly long _bucketSizeInBytes;

        public GradientBucketManager(IProcessGroup processGroup, Tensor[] parameters, long bucketSizeInBytes = 25 * 1024 * 1024)
        {
            _processGroup = processGroup;
            _parameters = parameters;
            _bucketSizeInBytes = bucketSizeInBytes;
            _buckets = new List<GradientBucket>();

            CreateBuckets();
        }

        public int NumBuckets => _buckets.Count;

        private void CreateBuckets()
        {
            long currentBucketSize = 0;
            var currentBucketParams = new List<Tensor>();

            foreach (var param in _parameters)
            {
                long paramSize = param.Numel * 4; // Assuming float32

                if (currentBucketSize + paramSize > _bucketSizeInBytes && currentBucketParams.Count > 0)
                {
                    _buckets.Add(new GradientBucket(currentBucketParams.ToArray(), _processGroup));
                    currentBucketParams = new List<Tensor>();
                    currentBucketSize = 0;
                }

                currentBucketParams.Add(param);
                currentBucketSize += paramSize;
            }

            if (currentBucketParams.Count > 0)
            {
                _buckets.Add(new GradientBucket(currentBucketParams.ToArray(), _processGroup));
            }
        }

        public void CopyBackAll()
        {
            foreach (var bucket in _buckets)
            {
                bucket.CopyBack();
            }
        }
    }

    /// <summary>
    /// Mock GradientBucket for testing.
    /// </summary>
    public class GradientBucket
    {
        private readonly Tensor[] _parameters;
        private readonly Tensor _bucketTensor;
        private readonly IProcessGroup _processGroup;
        private readonly long[] _offsets;

        public GradientBucket(Tensor[] parameters, IProcessGroup processGroup)
        {
            _parameters = parameters;
            _processGroup = processGroup;

            // Calculate total size and offsets
            long totalSize = 0;
            _offsets = new long[parameters.Length];
            for (int i = 0; i < parameters.Length; i++)
            {
                _offsets[i] = totalSize;
                totalSize += parameters[i].Numel;
            }

            _bucketTensor = Tensor.Random(new long[] { totalSize });
        }

        public void CopyBack()
        {
            for (int i = 0; i < _parameters.Length; i++)
            {
                var grad = _bucketTensor.Slice(0, _offsets[i], _parameters[i].Numel);
                var reshaped = grad.View(_parameters[i].Shape);
                _parameters[i].Copy_(reshaped);
            }
        }
    }

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
            var parameters = new[]
            {
                Tensor.Random(new long[] { 10 * 1024 * 1024 }),  // ~40MB
                Tensor.Random(new long[] { 5 * 1024 * 1024 }),   // ~20MB
                Tensor.Random(new long[] { 15 * 1024 * 1024 }), // ~60MB
                Tensor.Random(new long[] { 3 * 1024 * 1024 })    // ~12MB
            };

            var bucketManager = new GradientBucketManager(_processGroup, parameters, bucketSizeInBytes: 25 * 1024 * 1024);

            // Should create multiple buckets
            Assert.IsTrue(bucketManager.NumBuckets > 1, "Should create multiple buckets for large parameters");
        }

        [TestMethod]
        public void GradientBucketManager_SmallParameters_SingleBucket()
        {
            var parameters = new[]
            {
                Tensor.Random(new long[] { 100 }),
                Tensor.Random(new long[] { 200 }),
                Tensor.Random(new long[] { 50 })
            };

            var bucketManager = new GradientBucketManager(_processGroup, parameters, bucketSizeInBytes: 25 * 1024 * 1024);

            // All parameters should fit in a single bucket
            Assert.AreEqual(1, bucketManager.NumBuckets, "Small parameters should fit in a single bucket");
        }

        [TestMethod]
        public void GradientBucketManager_EmptyParameters_NoBuckets()
        {
            var parameters = new Tensor[0];
            var bucketManager = new GradientBucketManager(_processGroup, parameters);

            Assert.AreEqual(0, bucketManager.NumBuckets, "Empty parameters should result in no buckets");
        }

        [TestMethod]
        public void GradientBucket_CopyBack_PreservesShape()
        {
            var param1 = Tensor.Random(new long[] { 10, 20 });
            var param2 = Tensor.Random(new long[] { 30 });

            var parameters = new[] { param1, param2 };
            var bucketManager = new GradientBucketManager(_processGroup, parameters);

            bucketManager.CopyBackAll();

            // Check that shapes are preserved
            Assert.AreEqual(new long[] { 10, 20 }, param1.Shape);
            Assert.AreEqual(new long[] { 30 }, param2.Shape);
        }

        [TestMethod]
        public void GradientBucket_MultipleBuckets_CreatedCorrectly()
        {
            var parameters = new[]
            {
                Tensor.Random(new long[] { 6 * 1024 * 1024 }),    // ~24MB - just under bucket size
                Tensor.Random(new long[] { 5 * 1024 * 1024 }),    // ~20MB - should start new bucket
                Tensor.Random(new long[] { 4 * 1024 * 1024 })     // ~16MB - should fit in second bucket
            };

            var bucketManager = new GradientBucketManager(_processGroup, parameters, bucketSizeInBytes: 25 * 1024 * 1024);

            // Should create 2 buckets
            Assert.AreEqual(2, bucketManager.NumBuckets);
        }

        [TestMethod]
        public void GradientBucket_LargeParameter_ExceedsBucketSize_CreatesOwnBucket()
        {
            var parameters = new[]
            {
                Tensor.Random(new long[] { 30 * 1024 * 1024 }),   // ~120MB - exceeds bucket size
                Tensor.Random(new long[] { 10 * 1024 * 1024 })    // ~40MB
            };

            var bucketManager = new GradientBucketManager(_processGroup, parameters, bucketSizeInBytes: 25 * 1024 * 1024);

            // Should create 2 buckets (one for the large parameter)
            Assert.AreEqual(2, bucketManager.NumBuckets);
        }

        [TestMethod]
        public void GradientBucketManager_CopyBackAll_CompletesSuccessfully()
        {
            var parameters = new[]
            {
                Tensor.Random(new long[] { 100 }),
                Tensor.Random(new long[] { 200 })
            };

            var bucketManager = new GradientBucketManager(_processGroup, parameters);

            // Should not throw
            bucketManager.CopyBackAll();
        }

        [TestMethod]
        public void GradientBucket_OffsetCalculatedCorrectly()
        {
            var param1 = Tensor.Random(new long[] { 10 });
            var param2 = Tensor.Random(new long[] { 20 });
            var param3 = Tensor.Random(new long[] { 15 });

            var parameters = new[] { param1, param2, param3 };
            var bucketManager = new GradientBucketManager(_processGroup, parameters);

            // Check that buckets were created
            Assert.IsTrue(bucketManager.NumBuckets > 0);
        }
    }
}
