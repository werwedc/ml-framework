using MLFramework.NN;
using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLFramework.Distributed
{
    /// <summary>
    /// Manages gradient bucketing and asynchronous reduction.
    /// </summary>
    public class GradientBucketManager : IDisposable
    {
        private readonly IProcessGroup _processGroup;
        private readonly long _bucketSizeInBytes;
        private readonly Dictionary<Tensor, int> _tensorToBucketMap;
        private readonly GradientBucket[] _buckets;
        private bool _disposed;

        /// <summary>
        /// Creates a new gradient bucket manager.
        /// </summary>
        public GradientBucketManager(
            IProcessGroup processGroup,
            IEnumerable<Tensor> gradients,
            long bucketSizeInBytes = 25 * 1024 * 1024) // 25 MB default
        {
            _processGroup = processGroup ?? throw new ArgumentNullException(nameof(processGroup));
            _bucketSizeInBytes = bucketSizeInBytes;

            if (gradients == null)
            {
                throw new ArgumentNullException(nameof(gradients));
            }

            var gradientList = gradients.ToList();
            if (gradientList.Count == 0)
            {
                throw new ArgumentException("At least one gradient is required", nameof(gradients));
            }

            _tensorToBucketMap = new Dictionary<Tensor, int>();
            _buckets = CreateBuckets(gradientList, bucketSizeInBytes);
        }

        /// <summary>
        /// Gets the bucket index for a given gradient tensor.
        /// </summary>
        public int GetBucketIndex(Tensor gradient)
        {
            if (gradient == null)
            {
                throw new ArgumentNullException(nameof(gradient));
            }

            if (!_tensorToBucketMap.TryGetValue(gradient, out int bucketIndex))
            {
                throw new ArgumentException("Gradient is not managed by this bucket manager", nameof(gradient));
            }

            return bucketIndex;
        }

        /// <summary>
        /// Reduces a specific bucket asynchronously.
        /// </summary>
        public Task ReduceBucketAsync(int bucketIndex, ReduceOp op = ReduceOp.Sum)
        {
            if (bucketIndex < 0 || bucketIndex >= _buckets.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(bucketIndex));
            }

            return _buckets[bucketIndex].ReduceAsync(_processGroup, op);
        }

        /// <summary>
        /// Reduces all buckets asynchronously.
        /// </summary>
        public async Task ReduceAllAsync(ReduceOp op = ReduceOp.Sum)
        {
            var tasks = new Task[_buckets.Length];
            for (int i = 0; i < _buckets.Length; i++)
            {
                _buckets[i].Prepare();
                tasks[i] = _buckets[i].ReduceAsync(_processGroup, op);
            }
            await Task.WhenAll(tasks);
        }

        /// <summary>
        /// Waits for all bucket reductions to complete.
        /// </summary>
        public Task WaitForAllAsync()
        {
            var tasks = _buckets.Select(b => b.ReductionTask).ToArray();
            return Task.WhenAll(tasks);
        }

        /// <summary>
        /// Copies reduced values back to all gradient tensors.
        /// </summary>
        public void CopyBackAll()
        {
            foreach (var bucket in _buckets)
            {
                bucket.CopyBack();
            }
        }

        /// <summary>
        /// Creates buckets from the given gradients.
        /// </summary>
        private GradientBucket[] CreateBuckets(List<Tensor> gradients, long bucketSize)
        {
            var buckets = new List<GradientBucket>();
            var currentBucketGradients = new List<Tensor>();
            long currentBucketSize = 0;

            // Sort gradients by size (largest first helps with balancing)
            var sortedGradients = gradients.OrderByDescending(g => g.Size * sizeof(float)).ToList();

            int bucketIndex = 0;

            foreach (var gradient in sortedGradients)
            {
                long gradientSize = gradient.Size * sizeof(float);

                // If gradient is larger than bucket size, put it in its own bucket
                if (gradientSize > bucketSize)
                {
                    if (currentBucketGradients.Count > 0)
                    {
                        // Create current bucket before starting a new one
                        buckets.Add(new GradientBucket(bucketIndex++, currentBucketSize, currentBucketGradients.ToArray()));
                        currentBucketGradients.Clear();
                        currentBucketSize = 0;
                    }

                    // Create bucket for this large gradient
                    buckets.Add(new GradientBucket(bucketIndex++, gradientSize, new[] { gradient }));
                    _tensorToBucketMap[gradient] = bucketIndex - 1;
                    continue;
                }

                // Check if we should start a new bucket
                if (currentBucketSize + gradientSize > bucketSize && currentBucketGradients.Count > 0)
                {
                    // Create current bucket
                    buckets.Add(new GradientBucket(bucketIndex++, currentBucketSize, currentBucketGradients.ToArray()));

                    // Start new bucket
                    currentBucketGradients.Clear();
                    currentBucketSize = 0;
                }

                // Add gradient to current bucket
                currentBucketGradients.Add(gradient);
                currentBucketSize += gradientSize;
                _tensorToBucketMap[gradient] = bucketIndex;
            }

            // Create final bucket if it has gradients
            if (currentBucketGradients.Count > 0)
            {
                buckets.Add(new GradientBucket(bucketIndex, currentBucketSize, currentBucketGradients.ToArray()));
            }

            return buckets.ToArray();
        }

        /// <summary>
        /// Disposes the bucket manager.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Clean up managed resources
                    _tensorToBucketMap.Clear();
                }

                _disposed = true;
            }
        }

        ~GradientBucketManager()
        {
            Dispose(false);
        }
    }
}
