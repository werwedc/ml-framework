using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.NN;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// Represents a single stage in a pipeline parallel training setup.
    /// Wraps a module with metadata about its position in the pipeline.
    /// </summary>
    public class PipelineStage : Module
    {
        private readonly Module _module;

        /// <summary>
        /// Rank of this stage in the pipeline (0 to TotalStages-1)
        /// </summary>
        public int Rank { get; }

        /// <summary>
        /// Total number of pipeline stages
        /// </summary>
        public int TotalStages { get; }

        /// <summary>
        /// Device this stage executes on
        /// </summary>
        public IDevice Device { get; }

        /// <summary>
        /// The module containing the layers for this stage
        /// </summary>
        public Module Module => _module;

        /// <summary>
        /// Whether this is the first stage (receives input data)
        /// </summary>
        public bool IsFirstStage => Rank == 0;

        /// <summary>
        /// Whether this is the last stage (produces output)
        /// </summary>
        public bool IsLastStage => Rank == TotalStages - 1;

        /// <summary>
        /// Creates a new pipeline stage
        /// </summary>
        /// <param name="module">The module containing the layers for this stage</param>
        /// <param name="rank">Rank of this stage in the pipeline (0 to TotalStages-1)</param>
        /// <param name="totalStages">Total number of pipeline stages</param>
        /// <param name="device">Device this stage executes on</param>
        public PipelineStage(Module module, int rank, int totalStages, IDevice device)
            : base($"PipelineStage_{rank}")
        {
            _module = module ?? throw new ArgumentNullException(nameof(module));
            Device = device ?? throw new ArgumentNullException(nameof(device));

            // Validate inputs
            if (totalStages <= 0)
            {
                throw new ArgumentException("TotalStages must be greater than 0", nameof(totalStages));
            }

            if (rank < 0 || rank >= totalStages)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(rank),
                    $"Rank must be in range [0, {totalStages - 1}]");
            }

            Rank = rank;
            TotalStages = totalStages;

            // Move the provided module to the specified device
            MoveModuleToDevice(_module, Device);
        }

        /// <summary>
        /// Forward pass - delegates to the inner module
        /// </summary>
        public override Tensor Forward(Tensor input)
        {
            return _module.Forward(input);
        }

        /// <summary>
        /// Gets all trainable parameters of this stage (delegates to wrapped module)
        /// </summary>
        public override IEnumerable<Parameter> GetParameters()
        {
            return _module.GetParameters();
        }

        /// <summary>
        /// Gets all named parameters of this stage (delegates to wrapped module)
        /// </summary>
        public override IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters()
        {
            return _module.GetNamedParameters();
        }

        /// <summary>
        /// Enables or disables gradient computation for all parameters
        /// </summary>
        public override void SetRequiresGrad(bool requiresGrad)
        {
            _module.SetRequiresGrad(requiresGrad);
        }

        /// <summary>
        /// Moves a module's parameters to the specified device
        /// </summary>
        private void MoveModuleToDevice(Module module, IDevice device)
        {
            foreach (var param in module.GetParameters())
            {
                MoveParameterToDevice(param, device);
            }
        }

        /// <summary>
        /// Moves a parameter to the specified device
        /// </summary>
        private void MoveParameterToDevice(Parameter parameter, IDevice device)
        {
            // Note: This is a simplified implementation.
            // Parameter extends Tensor, so we need to handle device transfer.
            // The To() extension method works on Tensor, so we can use it directly on the parameter.
            // However, since Parameter is a reference type and we can't replace the actual Parameter object,
            // we assume that the Tensor.To() method will handle the device transfer internally.

            // For a production implementation, you would need to:
            // 1. Update the Parameter's internal device tracking
            // 2. Ensure all operations use device-specific implementations
            // 3. Handle memory management properly

            // For now, we call To() to ensure the tensor data is moved to the correct device
            var _ = parameter.To(device);
        }
    }
}
