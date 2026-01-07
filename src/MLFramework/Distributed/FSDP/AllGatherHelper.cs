using RitterFramework.Core.Tensor;
using MLFramework.Distributed;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Helper class for All-Gather operations in FSDP.
    /// </summary>
    public static class AllGatherHelper
    {
        /// <summary>
        /// Perform All-Gather on multiple tensors in parallel.
        /// </summary>
        /// <param name="processGroup">Process group for communication</param>
        /// <param name="tensors">Tensors to gather</param>
        /// <returns>List of gathered tensors</returns>
        public static Task<List<Tensor>> AllGatherMultipleAsync(IProcessGroup processGroup, List<Tensor> tensors)
        {
            if (tensors == null || tensors.Count == 0)
                return Task.FromResult(new List<Tensor>());

            var tasks = tensors.Select(tensor =>
            {
                var fullShape = tensor.Shape.Select(dim => (long)dim).ToArray();
                var op = new AllGatherOperation(processGroup, fullShape, tensor.Dtype, processGroup.Rank);
                return op.AllGatherAsync(tensor);
            }).ToList();

            return Task.WhenAll(tasks).ContinueWith(t => t.Result.ToList());
        }

        /// <summary>
        /// Calculate the optimal bucket size for All-Gather operations.
        /// </summary>
        /// <param name="worldSize">Number of devices</param>
        /// <param name="parameterSizes">Sizes of all parameters</param>
        /// <param name="maxBucketSizeMB">Maximum bucket size in MB</param>
        /// <returns>List of bucket indices for each parameter</returns>
        public static List<int> CalculateBuckets(int worldSize, List<long> parameterSizes, int maxBucketSizeMB)
        {
            var buckets = new List<int>();
            var currentBucketSize = 0L;
            var currentBucketIndex = 0;
            var maxBucketSizeBytes = (long)maxBucketSizeMB * 1024 * 1024;

            foreach (var paramSize in parameterSizes)
            {
                var fullSize = paramSize * worldSize; // Size after gathering

                if (currentBucketSize + fullSize > maxBucketSizeBytes)
                {
                    currentBucketIndex++;
                    currentBucketSize = 0;
                }

                buckets.Add(currentBucketIndex);
                currentBucketSize += fullSize;
            }

            return buckets;
        }
    }
}
