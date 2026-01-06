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
    public class FSDPOptimizerStateManager
    {
        private readonly IProcessGroup _processGroup;
        private readonly Dictionary<string, OptimizerState> _optimizerStates;

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

            // Multi-device case: gather to rank 0
            // Note: This will be implemented in a future spec for communication primitives
            throw new System.NotImplementedException("Multi-device optimizer state gathering to be implemented");
        }

        /// <summary>
        /// Remove an optimizer state from the manager.
        /// </summary>
        /// <param name="parameterName">Name of the parameter</param>
        /// <returns>True if the state was removed, false otherwise</returns>
        public bool RemoveOptimizerState(string parameterName)
        {
            return _optimizerStates.Remove(parameterName);
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
    }
}
