using System;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using MLFramework.Distributed;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Type of optimizer state being managed.
    /// </summary>
    public enum OptimizerStateType
    {
        /// <summary>No optimizer state</summary>
        None,

        /// <summary>SGD: just the parameter itself</summary>
        SGD,

        /// <summary>Adam: momentum and variance</summary>
        Adam,

        /// <summary>AdamW: Adam with weight decay</summary>
        AdamW
    }

    /// <summary>
    /// Manages a single sharded parameter for FSDP (Fully Sharded Data Parallelism).
    /// Handles sharding, gathering, and tracking of parameter state across distributed devices.
    /// </summary>
    public class FSDPShardingUnit : IDisposable
    {
        private readonly IProcessGroup _processGroup;
        private readonly FSDPState _state;
        private bool _disposed;

        /// <summary>Local shard of the parameter (only stores 1/num_devices)</summary>
        public Tensor? ShardedParameter { get; set; }

        /// <summary>Full gathered parameter (temporarily allocated during forward/backward)</summary>
        public Tensor? GatheredParameter { get; set; }

        /// <summary>Local gradient for this shard</summary>
        public Tensor? LocalGradient { get; set; }

        /// <summary>Optimizer state for this shard (momentum, variance, etc.)</summary>
        public object? LocalOptimizerState { get; set; }

        /// <summary>Original parameter name</summary>
        public string ParameterName { get; set; }

        /// <summary>Parameter shape</summary>
        public int[] Shape { get; set; }

        /// <summary>Parameter data type</summary>
        public DataType DataType { get; set; }

        /// <summary>
        /// Current state of this sharded unit.
        /// </summary>
        public FSDPState State => _state;

        /// <summary>
        /// Initialize a new sharding unit for a parameter.
        /// </summary>
        /// <param name="parameterName">Name of the parameter</param>
        /// <param name="fullParameter">Full parameter tensor (will be sharded)</param>
        /// <param name="processGroup">Process group for communication</param>
        /// <exception cref="ArgumentException">Thrown when parameter name is empty</exception>
        /// <exception cref="ArgumentNullException">Thrown when processGroup is null</exception>
        public FSDPShardingUnit(string parameterName, Tensor fullParameter, IProcessGroup processGroup)
        {
            if (string.IsNullOrEmpty(parameterName))
                throw new ArgumentException("Parameter name cannot be empty", nameof(parameterName));

            _processGroup = processGroup ?? throw new ArgumentNullException(nameof(processGroup));
            ParameterName = parameterName;

            // Copy shape and data type from the full parameter
            Shape = new int[fullParameter.Shape.Length];
            Array.Copy(fullParameter.Shape, Shape, fullParameter.Shape.Length);
            DataType = fullParameter.Dtype;

            // Calculate shard index and owner rank
            var worldSize = _processGroup.WorldSize;
            var rank = _processGroup.Rank;
            var numParameters = fullParameter.Size;
            var shardSize = (numParameters + worldSize - 1) / worldSize;

            // Determine which shard this rank owns
            var shardStart = rank * shardSize;
            var shardEnd = Math.Min(shardStart + shardSize, numParameters);
            var actualShardSize = shardEnd - shardStart;

            // Create local shard
            ShardedParameter = Tensor.Zeros(new[] { actualShardSize }, DataType);

            // Initialize state
            _state = new FSDPState
            {
                OwnerRank = rank,
                NumShards = worldSize,
                ShardIndex = rank,
                IsGathered = false,
                IsOffloaded = false
            };

            // Copy the portion of the parameter that belongs to this shard
            var flatData = fullParameter.Data;
            var shardData = ShardedParameter.Data;
            Array.Copy(flatData, shardStart, shardData, 0, actualShardSize);
        }

        /// <summary>
        /// Gather parameters from all devices.
        /// Creates a full-sized parameter tensor containing all shards.
        /// </summary>
        /// <returns>Full gathered parameter</returns>
        /// <exception cref="NotImplementedException">Thrown when communication primitives are not yet implemented</exception>
        public Tensor GatherParameters()
        {
            if (_state.IsGathered)
                return GatheredParameter!;

            var worldSize = _processGroup.WorldSize;
            var rank = _processGroup.Rank;

            // Allocate buffer for gathered parameter
            GatheredParameter = Tensor.Zeros(Shape, DataType);

            // All-gather: combine all shards into full parameter
            // This will be implemented in a separate spec for communication primitives
            throw new NotImplementedException("Communication primitives to be implemented in spec_fsdp_all_gather.md");
        }

        /// <summary>
        /// Release gathered parameters to free memory.
        /// </summary>
        public void ReleaseGatheredParameters()
        {
            GatheredParameter = null;
            _state.IsGathered = false;
        }

        /// <summary>
        /// Scatter gradients to owning devices.
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown when no gradient is available to scatter</exception>
        /// <exception cref="NotImplementedException">Thrown when communication primitives are not yet implemented</exception>
        public void ScatterGradients()
        {
            if (LocalGradient == null)
                throw new InvalidOperationException("No gradient to scatter");

            // Scatter gradients to owner ranks
            // This will be implemented in a separate spec for communication primitives
            throw new NotImplementedException("Communication primitives to be implemented in spec_fsdp_reduce_scatter.md");
        }

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
        /// <param name="disposing">Whether managed resources should be disposed</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Tensor class doesn't implement IDisposable, so we just set to null
                    ShardedParameter = null;
                    GatheredParameter = null;
                    LocalGradient = null;
                }

                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for FSDPShardingUnit.
        /// </summary>
        ~FSDPShardingUnit()
        {
            Dispose(false);
        }
    }
}
