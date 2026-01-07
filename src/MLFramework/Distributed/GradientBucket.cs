using MLFramework.Distributed;
using RitterFramework.Core.Tensor;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace MLFramework.Distributed
{
    /// <summary>
    /// Represents a bucket of gradients that will be reduced together.
    /// </summary>
    public class GradientBucket
    {
        private readonly Tensor[] _gradients;
        private readonly int[] _elementOffsets;
        private readonly int[] _elementSizes;
        private Tensor _bucketTensor;
        private Task _reductionTask;
        private bool _prepared;

        /// <summary>
        /// Gets the bucket index.
        /// </summary>
        public int BucketIndex { get; }

        /// <summary>
        /// Gets the size of the bucket in bytes.
        /// </summary>
        public long SizeInBytes { get; }

        /// <summary>
        /// Gets the original gradient tensors in this bucket.
        /// </summary>
        public Tensor[] Gradients => _gradients;

        /// <summary>
        /// Gets the flattened and concatenated bucket tensor.
        /// </summary>
        public Tensor BucketTensor => _bucketTensor;

        /// <summary>
        /// Gets the current reduction task.
        /// </summary>
        public Task ReductionTask => _reductionTask;

        /// <summary>
        /// Gets whether the bucket reduction is complete.
        /// </summary>
        public bool IsReduced => _reductionTask?.IsCompleted ?? false;

        /// <summary>
        /// Creates a new gradient bucket.
        /// </summary>
        public GradientBucket(int bucketIndex, long sizeInBytes, Tensor[] gradients)
        {
            BucketIndex = bucketIndex;
            SizeInBytes = sizeInBytes;
            _gradients = gradients ?? throw new ArgumentNullException(nameof(gradients));

            if (_gradients.Length == 0)
            {
                throw new ArgumentException("Bucket must contain at least one gradient", nameof(gradients));
            }

            // Calculate offsets and sizes for each gradient (in elements, not bytes)
            _elementOffsets = new int[gradients.Length];
            _elementSizes = new int[gradients.Length];

            int currentOffset = 0;
            for (int i = 0; i < gradients.Length; i++)
            {
                _elementOffsets[i] = currentOffset;
                _elementSizes[i] = gradients[i].Size;
                currentOffset += _elementSizes[i];
            }

            _bucketTensor = Tensor.Zeros(new int[] { (int)(sizeInBytes / sizeof(float)) });
            _reductionTask = Task.CompletedTask;
            _prepared = false;
        }

        /// <summary>
        /// Prepares the bucket for reduction by flattening and concatenating gradients.
        /// </summary>
        public void Prepare()
        {
            if (_prepared)
            {
                return;
            }

            // Copy each gradient into the bucket tensor at its offset
            float[] bucketData = _bucketTensor.Data;

            for (int i = 0; i < _gradients.Length; i++)
            {
                Array.Copy(
                    _gradients[i].Data,
                    0,
                    bucketData,
                    _elementOffsets[i],
                    _elementSizes[i]
                );
            }

            _prepared = true;
        }

        /// <summary>
        /// Reduces the bucket asynchronously using the process group.
        /// </summary>
        public async Task ReduceAsync(IProcessGroup processGroup, ReduceOp op = ReduceOp.Sum)
        {
            if (processGroup == null)
            {
                throw new ArgumentNullException(nameof(processGroup));
            }

            if (!_prepared)
            {
                Prepare();
            }

            _reductionTask = processGroup.AllReduceAsync(_bucketTensor, op);
            await _reductionTask;
        }

        /// <summary>
        /// Copies reduced values back to original gradient tensors.
        /// </summary>
        public void CopyBack()
        {
            if (!_prepared)
            {
                throw new InvalidOperationException("Cannot copy back before bucket is prepared");
            }

            if (_reductionTask.Status != TaskStatus.RanToCompletion)
            {
                throw new InvalidOperationException("Cannot copy back before reduction is complete");
            }

            float[] bucketData = _bucketTensor.Data;

            for (int i = 0; i < _gradients.Length; i++)
            {
                Array.Copy(
                    bucketData,
                    _elementOffsets[i],
                    _gradients[i].Data,
                    0,
                    _elementSizes[i]
                );
            }

            _prepared = false;
        }
    }
}
