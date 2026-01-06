using System;
using RitterFramework.Core.Tensor;

namespace MLFramework.Distributed
{
    /// <summary>
    /// Hook that is attached to parameters to trigger gradient reduction.
    /// </summary>
    internal class GradientSynchronizationHook
    {
        private readonly DistributedDataParallel _ddp;
        private readonly NN.Parameter _parameter;
        private readonly GradientBucketManager _bucketManager;

        /// <summary>
        /// Creates a new gradient synchronization hook.
        /// </summary>
        public GradientSynchronizationHook(
            DistributedDataParallel ddp,
            NN.Parameter parameter,
            GradientBucketManager bucketManager)
        {
            _ddp = ddp ?? throw new ArgumentNullException(nameof(ddp));
            _parameter = parameter ?? throw new ArgumentNullException(nameof(parameter));
            _bucketManager = bucketManager ?? throw new ArgumentNullException(nameof(bucketManager));
        }

        /// <summary>
        /// Called when a gradient is computed during backward pass.
        /// </summary>
        public Tensor OnGradient(Tensor gradient)
        {
            // Mark parameter as used (for findUnusedParameters mode)
            if (_ddp.FindUnusedParameters)
            {
                _ddp.MarkParameterAsUsed(_parameter.Name);
            }

            // Trigger bucket reduction asynchronously
            try
            {
                var bucketIndex = _bucketManager.GetBucketIndex(_parameter);
                _bucketManager.ReduceBucketAsync(bucketIndex, ReduceOp.Sum);
            }
            catch (ArgumentException)
            {
                // Parameter might not be in the bucket manager (e.g., frozen parameters)
                // This is okay, just skip
            }

            return gradient;
        }
    }
}
