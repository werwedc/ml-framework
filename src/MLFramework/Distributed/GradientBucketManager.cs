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
            IEnumerable<Parameter> parameters,
            long bucketSizeInBytes = 25 * 1024 * 1024) // 25 MB default
        {
            _processGroup = processGroup ?? throw new ArgumentNullException(nameof(processGroup));
            _bucketSizeInBytes = bucketSizeInBytes;

            if (parameters == null)
            {
                throw new ArgumentNullException(nameof(parameters));
            }

            var parameterList = parameters.ToList();
            if (parameterList.Count == 0)
            {
                throw new ArgumentException("At least one parameter is required", nameof(parameters));
            }

            _tensorToBucketMap = new Dictionary<Tensor, int>();
            _buckets = CreateBuckets(parameterList, bucketSizeInBytes);
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
        /// Creates buckets from the given parameters.
        /// </summary>
        private GradientBucket[] CreateBuckets(List<Parameter> parameters, long bucketSize)
        {
            var buckets = new List<GradientBucket>();
            var currentBucketParams = new List<Parameter>();
            long currentBucketSize = 0;

            // Sort parameters by size (largest first helps with balancing)
            var sortedParams = parameters.OrderByDescending(p => p.Size * sizeof(float)).ToList();

            int bucketIndex = 0;

            foreach (var param in sortedParams)
            {
                long paramSize = param.Size * sizeof(float);

                // If parameter is larger than bucket size, put it in its own bucket
                if (paramSize > bucketSize)
                {
                    if (currentBucketParams.Count > 0)
                    {
                        // Create current bucket before starting a new one
                        var gradients = currentBucketParams.Select(p => p).ToArray();
                        buckets.Add(new GradientBucket(bucketIndex++, currentBucketSize, gradients));
                        currentBucketParams.Clear();
                        currentBucketSize = 0;
                    }

                    // Create bucket for this large parameter
                    buckets.Add(new GradientBucket(bucketIndex++, paramSize, new[] { param }));
                    _tensorToBucketMap[param] = bucketIndex - 1;
                    continue;
                }

                // Check if we should start a new bucket
                if (currentBucketSize + paramSize > bucketSize && currentBucketParams.Count > 0)
                {
                    // Create current bucket
                    var gradients = currentBucketParams.Select(p => p).ToArray();
                    buckets.Add(new GradientBucket(bucketIndex++, currentBucketSize, gradients));

                    // Start new bucket
                    currentBucketParams.Clear();
                    currentBucketSize = 0;
                }

                // Add parameter to current bucket
                currentBucketParams.Add(param);
                currentBucketSize += paramSize;
                _tensorToBucketMap[param] = bucketIndex;
            }

            // Create final bucket if it has parameters
            if (currentBucketParams.Count > 0)
            {
                var gradients = currentBucketParams.Select(p => p).ToArray();
                buckets.Add(new GradientBucket(bucketIndex, currentBucketSize, gradients));
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
