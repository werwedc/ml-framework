using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Manages optimizer states for sharded parameters in FSDP.
    /// Handles gathering, scattering, and updating optimizer states across devices.
    /// </summary>
    public class FSDPOptimizerStateManager : IDisposable
    {
        private readonly IProcessGroup _processGroup;
        private readonly Dictionary<string, OptimizerState> _optimizerStates;
        private bool _disposed;

        /// <summary>
        /// Create a new optimizer state manager.
        /// </summary>
        /// <param name="processGroup">Process group for distributed communication</param>
        public FSDPOptimizerStateManager(IProcessGroup processGroup)
        {
            _processGroup = processGroup ?? throw new System.ArgumentNullException(nameof(processGroup));
            _optimizerStates = new Dictionary<string, OptimizerState>();
        }

        /// <summary>
        /// Create optimizer state for a sharded parameter.
        /// </summary>
        /// <param name="parameterName">Parameter name</param>
        /// <param name="shardedParameter">Sharded parameter tensor</param>
        /// <param name="optimizerType">Type of optimizer</param>
        /// <param name="shardIndex">Shard index</param>
        /// <param name="numShards">Total number of shards</param>
        /// <returns>Created optimizer state</returns>
        public OptimizerState CreateOptimizerState(
            string parameterName,
            Tensor shardedParameter,
            OptimizerStateType optimizerType,
            int shardIndex,
            int numShards)
        {
            if (string.IsNullOrEmpty(parameterName))
                throw new System.ArgumentException("Parameter name cannot be empty", nameof(parameterName));

            if (shardedParameter == null)
                throw new System.ArgumentNullException(nameof(shardedParameter));

            if (_optimizerStates.ContainsKey(parameterName))
                throw new System.ArgumentException($"Optimizer state already exists for {parameterName}");

            OptimizerState state = optimizerType switch
            {
                OptimizerStateType.Adam => new AdamOptimizerState(shardedParameter, shardIndex, numShards),
                OptimizerStateType.AdamW => new AdamWOptimizerState(shardedParameter, shardIndex, numShards),
                OptimizerStateType.SGD => new SGDOptimizerState(shardIndex, numShards),
                _ => throw new System.ArgumentException($"Unsupported optimizer type: {optimizerType}", nameof(optimizerType))
            };

            _optimizerStates[parameterName] = state;
            return state;
        }

        /// <summary>
        /// Register an optimizer state for a parameter.
        /// </summary>
        /// <param name="parameterName">Name of the parameter</param>
        /// <param name="state">Optimizer state to register</param>
        public void RegisterOptimizerState(string parameterName, OptimizerState state)
        {
            if (string.IsNullOrEmpty(parameterName))
                throw new System.ArgumentException("Parameter name cannot be empty", nameof(parameterName));

            if (state == null)
                throw new System.ArgumentNullException(nameof(state));

            _optimizerStates[parameterName] = state;
        }

        /// <summary>
        /// Check if optimizer state exists for a parameter.
        /// </summary>
        /// <param name="parameterName">Parameter name</param>
        /// <returns>True if state exists</returns>
        public bool HasOptimizerState(string parameterName)
        {
            return _optimizerStates.ContainsKey(parameterName);
        }

        /// <summary>
        /// Get the optimizer state for a parameter.
        /// </summary>
        /// <param name="parameterName">Name of the parameter</param>
        /// <returns>Optimizer state for the parameter</returns>
        public OptimizerState GetOptimizerState(string parameterName)
        {
            if (_optimizerStates.TryGetValue(parameterName, out var state))
            {
                return state;
            }
            return null;
        }

        /// <summary>
        /// Get all parameter names with registered optimizer states.
        /// </summary>
        /// <returns>List of parameter names</returns>
        public List<string> GetAllParameterNames()
        {
            return _optimizerStates.Keys.ToList();
        }

        /// <summary>
        /// Gather optimizer state from all ranks to the current rank.
        /// Only rank 0 will receive the full gathered state.
        /// </summary>
        /// <param name="parameterName">Name of the parameter</param>
        /// <returns>Gathered optimizer state (only on rank 0)</returns>
        public OptimizerState GatherOptimizerState(string parameterName)
        {
            if (!_optimizerStates.TryGetValue(parameterName, out var localState))
            {
                return null;
            }

            // For single device case, just return a clone
            if (_processGroup.WorldSize == 1)
            {
                return localState.Clone();
            }

            // Gather all shards of optimizer state
            if (localState is AdamOptimizerState adamState)
            {
                return GatherAdamState(parameterName, adamState);
            }
            else if (localState is SGDOptimizerState sgdState)
            {
                // SGD has no state to gather
                return sgdState.Clone();
            }

            throw new System.ArgumentException($"Unsupported optimizer state type: {localState.StateType}");
        }

        /// <summary>
        /// Scatter optimizer state to devices (for loading checkpoints).
        /// </summary>
        /// <param name="parameterName">Parameter name</param>
        /// <param name="fullState">Full optimizer state from checkpoint</param>
        public void ScatterOptimizerState(string parameterName, OptimizerState fullState)
        {
            if (fullState == null)
                throw new System.ArgumentNullException(nameof(fullState));

            if (!_optimizerStates.TryGetValue(parameterName, out var localState))
                throw new System.ArgumentException($"No optimizer state found for {parameterName}");

            if (_processGroup.WorldSize == 1)
            {
                // Single device, just copy
                CopyOptimizerState(fullState, localState);
                return;
            }

            // Broadcast from rank 0 and extract local shard
            if (fullState is AdamOptimizerState fullAdamState && localState is AdamOptimizerState localAdamState)
            {
                ScatterAdamState(parameterName, fullAdamState, localAdamState);
            }
        }

        /// <summary>
        /// Gather Adam optimizer state from all devices.
        /// </summary>
        private AdamOptimizerState GatherAdamState(string parameterName, AdamOptimizerState localState)
        {
            var worldSize = _processGroup.WorldSize;
            var rank = _processGroup.Rank;

            // Note: This requires AllGatherOperation which should be available
            // For now, we'll implement a basic version that works with available interfaces
            // The full implementation would use AllGatherOperation

            // Since AllGatherOperation exists in FSDP, we can use it
            // Convert int[] to long[] for AllGatherOperation
            var momentumShape = new long[localState.MomentumBuffer.Shape.Length];
            for (int i = 0; i < localState.MomentumBuffer.Shape.Length; i++)
            {
                momentumShape[i] = localState.MomentumBuffer.Shape[i];
            }

            var varianceShape = new long[localState.VarianceBuffer.Shape.Length];
            for (int i = 0; i < localState.VarianceBuffer.Shape.Length; i++)
            {
                varianceShape[i] = localState.VarianceBuffer.Shape[i];
            }

            var allGatherOp = new AllGatherOperation(_processGroup, momentumShape, localState.MomentumBuffer.Dtype, rank);
            var fullMomentum = allGatherOp.AllGather(localState.MomentumBuffer);

            var allGatherOp2 = new AllGatherOperation(_processGroup, varianceShape, localState.VarianceBuffer.Dtype, rank);
            var fullVariance = allGatherOp2.AllGather(localState.VarianceBuffer);

            // Create full state (only rank 0 returns the full state)
            AdamOptimizerState fullState;
            if (rank == 0)
            {
                fullState = new AdamOptimizerState(fullMomentum, 0, worldSize);
                fullState.MomentumBuffer = fullMomentum;
                fullState.VarianceBuffer = fullVariance;
                fullState.StepCount = localState.StepCount;
            }
            else
            {
                fullState = null;
            }

            return fullState;
        }

        /// <summary>
        /// Scatter Adam optimizer state to devices.
        /// </summary>
        private void ScatterAdamState(string parameterName, AdamOptimizerState fullState, AdamOptimizerState localState)
        {
            var worldSize = _processGroup.WorldSize;
            var rank = _processGroup.Rank;

            // Broadcast full momentum buffer
            _processGroup.Broadcast(fullState.MomentumBuffer, 0);

            // Broadcast full variance buffer
            _processGroup.Broadcast(fullState.VarianceBuffer, 0);

            // Extract local shard from full state
            var shardSize = localState.MomentumBuffer.Size;
            var shardOffset = rank * shardSize;

            System.Array.Copy(fullState.MomentumBuffer.Data, shardOffset, localState.MomentumBuffer.Data, 0, shardSize);
            System.Array.Copy(fullState.VarianceBuffer.Data, shardOffset, localState.VarianceBuffer.Data, 0, shardSize);

            localState.StepCount = fullState.StepCount;
        }

        /// <summary>
        /// Copy optimizer state from source to destination.
        /// </summary>
        private void CopyOptimizerState(OptimizerState source, OptimizerState destination)
        {
            if (source.StateType != destination.StateType)
                throw new System.ArgumentException("Optimizer state types must match");

            if (source is AdamOptimizerState sourceAdam && destination is AdamOptimizerState destAdam)
            {
                System.Array.Copy(sourceAdam.MomentumBuffer.Data, destAdam.MomentumBuffer.Data, sourceAdam.MomentumBuffer.Size);
                System.Array.Copy(sourceAdam.VarianceBuffer.Data, destAdam.VarianceBuffer.Data, sourceAdam.VarianceBuffer.Size);
                destAdam.StepCount = sourceAdam.StepCount;
            }
        }

        /// <summary>
        /// Remove an optimizer state from the manager.
        /// </summary>
        /// <param name="parameterName">Name of the parameter</param>
        public void RemoveOptimizerState(string parameterName)
        {
            if (_optimizerStates.TryGetValue(parameterName, out var state))
            {
                state.Dispose();
                _optimizerStates.Remove(parameterName);
            }
        }

        /// <summary>
        /// Clear all optimizer states.
        /// </summary>
        public void ClearAll()
        {
            _optimizerStates.Clear();
        }

        /// <summary>
        /// Get the number of registered optimizer states.
        /// </summary>
        /// <returns>Number of states</returns>
        public int Count => _optimizerStates.Count;

        /// <summary>
        /// Dispose of resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            System.GC.SuppressFinalize(this);
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
                    foreach (var state in _optimizerStates.Values)
                    {
                        state.Dispose();
                    }
                    _optimizerStates.Clear();
                }
                _disposed = true;
            }
        }
    }
}
