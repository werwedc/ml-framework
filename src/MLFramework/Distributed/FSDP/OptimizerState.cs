using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using System;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Base class for optimizer state management in FSDP.
    /// </summary>
    public abstract class OptimizerState : IDisposable
    {
        /// <summary>Type of optimizer state</summary>
        public OptimizerStateType StateType { get; protected set; }

        /// <summary>Index of this shard</summary>
        public int ShardIndex { get; protected set; }

        /// <summary>Total number of shards</summary>
        public int NumShards { get; protected set; }

        /// <summary>Number of optimization steps</summary>
        public int StepCount { get; set; }

        /// <summary>Whether this state has been disposed</summary>
        protected bool _disposed;

        /// <summary>
        /// Create base optimizer state.
        /// </summary>
        protected OptimizerState(int shardIndex, int numShards, OptimizerStateType stateType)
        {
            ShardIndex = shardIndex;
            NumShards = numShards;
            StateType = stateType;
            StepCount = 0;
        }

        /// <summary>
        /// Clone this optimizer state.
        /// </summary>
        /// <returns>Cloned state</returns>
        public abstract OptimizerState Clone();

        /// <summary>
        /// Dispose of resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Protected implementation of dispose pattern.
        /// </summary>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// SGD optimizer state (minimal state).
    /// </summary>
    public class SGDOptimizerState : OptimizerState
    {
        /// <summary>
        /// Create SGD optimizer state.
        /// </summary>
        /// <param name="shardIndex">Index of this shard</param>
        /// <param name="numShards">Total number of shards</param>
        public SGDOptimizerState(int shardIndex, int numShards)
            : base(shardIndex, numShards, OptimizerStateType.SGD)
        {
        }

        /// <inheritdoc/>
        public override OptimizerState Clone()
        {
            var cloned = new SGDOptimizerState(ShardIndex, NumShards);
            cloned.StepCount = StepCount;
            return cloned;
        }
    }

    /// <summary>
    /// Adam optimizer state with momentum and variance buffers.
    /// </summary>
    public class AdamOptimizerState : OptimizerState
    {
        /// <summary>Momentum buffer (first moment estimate)</summary>
        public Tensor? MomentumBuffer { get; set; }

        /// <summary>Variance buffer (second moment estimate)</summary>
        public Tensor? VarianceBuffer { get; set; }

        /// <summary>
        /// Create Adam optimizer state.
        /// </summary>
        /// <param name="parameter">Parameter tensor to allocate buffers for</param>
        /// <param name="shardIndex">Index of this shard</param>
        /// <param name="numShards">Total number of shards</param>
        public AdamOptimizerState(Tensor parameter, int shardIndex, int numShards)
            : base(shardIndex, numShards, OptimizerStateType.Adam)
        {
            if (parameter == null)
                throw new ArgumentNullException(nameof(parameter));

            // Allocate buffers with same shape as parameter
            MomentumBuffer = Tensor.Zeros(parameter.Shape, parameter.Dtype);
            VarianceBuffer = Tensor.Zeros(parameter.Shape, parameter.Dtype);
        }

        /// <summary>
        /// Create Adam optimizer state with explicit buffer sizes.
        /// </summary>
        /// <param name="bufferSize">Size of buffers</param>
        /// <param name="dataType">Data type of buffers</param>
        /// <param name="shardIndex">Index of this shard</param>
        /// <param name="numShards">Total number of shards</param>
        public AdamOptimizerState(long bufferSize, DataType dataType, int shardIndex, int numShards)
            : base(shardIndex, numShards, OptimizerStateType.Adam)
        {
            MomentumBuffer = Tensor.Zeros(new[] { (int)bufferSize }, dataType);
            VarianceBuffer = Tensor.Zeros(new[] { (int)bufferSize }, dataType);
        }

        /// <inheritdoc/>
        public override OptimizerState Clone()
        {
            var cloned = new AdamOptimizerState(
                MomentumBuffer?.Size ?? 0,
                MomentumBuffer?.Dtype ?? DataType.Float32,
                ShardIndex,
                NumShards
            );

            cloned.StepCount = StepCount;

            // Clone the buffers
            if (MomentumBuffer != null)
            {
                cloned.MomentumBuffer = MomentumBuffer.Clone();
            }

            if (VarianceBuffer != null)
            {
                cloned.VarianceBuffer = VarianceBuffer.Clone();
            }

            return cloned;
        }

        /// <inheritdoc/>
        protected override void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources
                    MomentumBuffer = null;
                    VarianceBuffer = null;
                }
                base.Dispose(disposing);
            }
        }
    }

    /// <summary>
    /// AdamW optimizer state (Adam with weight decay).
    /// </summary>
    public class AdamWOptimizerState : AdamOptimizerState
    {
        /// <summary>
        /// Create AdamW optimizer state.
        /// </summary>
        /// <param name="parameter">Parameter tensor to allocate buffers for</param>
        /// <param name="shardIndex">Index of this shard</param>
        /// <param name="numShards">Total number of shards</param>
        public AdamWOptimizerState(Tensor parameter, int shardIndex, int numShards)
            : base(parameter, shardIndex, numShards)
        {
            StateType = OptimizerStateType.AdamW;
        }

        /// <summary>
        /// Create AdamW optimizer state with explicit buffer sizes.
        /// </summary>
        /// <param name="bufferSize">Size of buffers</param>
        /// <param name="dataType">Data type of buffers</param>
        /// <param name="shardIndex">Index of this shard</param>
        /// <param name="numShards">Total number of shards</param>
        public AdamWOptimizerState(long bufferSize, DataType dataType, int shardIndex, int numShards)
            : base(bufferSize, dataType, shardIndex, numShards)
        {
            StateType = OptimizerStateType.AdamW;
        }

        /// <inheritdoc/>
        public override OptimizerState Clone()
        {
            var cloned = base.Clone() as AdamWOptimizerState;
            if (cloned != null)
            {
                cloned.StateType = OptimizerStateType.AdamW;
            }
            return cloned!;
        }
    }
}
