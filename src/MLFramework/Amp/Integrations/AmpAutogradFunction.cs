using System;
using System.Linq;
using MLFramework.Core;
using MLFramework.Amp;
using RitterFramework.Core.Tensor;

namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// Base class for AMP-aware autograd functions
    /// </summary>
    public abstract class AmpAutogradFunction
    {
        /// <summary>
        /// Gets the operation precision registry
        /// </summary>
        protected AmpRegistry? Registry { get; }

        /// <summary>
        /// Gets the AutoCast mode
        /// </summary>
        protected AutoCastMode Mode { get; }

        /// <summary>
        /// Creates a new AmpAutogradFunction
        /// </summary>
        protected AmpAutogradFunction(AmpRegistry? registry = null)
        {
            Registry = registry;
            Mode = registry != null ? MapDataTypeToAutoCastMode(registry.GetConfig().TargetPrecision) : AutoCastMode.Bf16;
        }

        private AutoCastMode MapDataTypeToAutoCastMode(DataType dtype)
        {
            return dtype switch
            {
                DataType.Float16 => AutoCastMode.Fp16,
                DataType.BFloat16 => AutoCastMode.Bf16,
                _ => AutoCastMode.None
            };
        }

        /// <summary>
        /// Forward pass with automatic precision casting
        /// </summary>
        /// <param name="inputs">Input tensors</param>
        /// <param name="operationType">The type of operation</param>
        /// <returns>Output tensors</returns>
        public Tensor[] Forward(Tensor[] inputs, Type operationType)
        {
            if (inputs == null || inputs.Length == 0)
                throw new ArgumentException("Inputs cannot be null or empty", nameof(inputs));

            if (operationType == null)
                throw new ArgumentNullException(nameof(operationType));

            var inputDtype = inputs[0].GetDtype();
            var forwardDtype = GetForwardDtype(operationType, inputDtype);
            var backwardDtype = GetBackwardDtype(operationType, inputDtype);

            return ForwardManual(inputs, forwardDtype, backwardDtype);
        }

        /// <summary>
        /// Forward pass with manual precision specification
        /// </summary>
        /// <param name="inputs">Input tensors</param>
        /// <param name="forwardDtype">The dtype for forward pass</param>
        /// <param name="backwardDtype">The dtype for backward pass</param>
        /// <returns>Output tensors</returns>
        public abstract Tensor[] ForwardManual(
            Tensor[] inputs,
            DataType forwardDtype,
            DataType backwardDtype);

        /// <summary>
        /// Backward pass with automatic precision handling
        /// </summary>
        /// <param name="gradOutputs">Gradient outputs</param>
        /// <param name="operationType">The type of operation</param>
        /// <returns>Gradient inputs</returns>
        public Tensor[] Backward(Tensor[] gradOutputs, Type operationType)
        {
            if (gradOutputs == null || gradOutputs.Length == 0)
                throw new ArgumentException("Gradient outputs cannot be null or empty", nameof(gradOutputs));

            if (operationType == null)
                throw new ArgumentNullException(nameof(operationType));

            var outputDtype = gradOutputs[0].GetDtype();
            var backwardDtype = GetBackwardDtype(operationType, outputDtype);

            return BackwardManual(gradOutputs, backwardDtype);
        }

        /// <summary>
        /// Backward pass with manual precision specification
        /// </summary>
        /// <param name="gradOutputs">Gradient outputs</param>
        /// <param name="backwardDtype">The dtype for backward pass</param>
        /// <returns>Gradient inputs</returns>
        public abstract Tensor[] BackwardManual(
            Tensor[] gradOutputs,
            DataType backwardDtype);

        /// <summary>
        /// Gets the forward dtype for an operation
        /// </summary>
        protected DataType GetForwardDtype(Type operationType, DataType inputDtype)
        {
            if (Registry != null)
            {
                return Registry.GetForwardDtype(operationType, inputDtype);
            }

            // Default policy: use input dtype
            return inputDtype;
        }

        /// <summary>
        /// Gets the backward dtype for an operation
        /// </summary>
        protected DataType GetBackwardDtype(Type operationType, DataType inputDtype)
        {
            if (Registry != null)
            {
                return Registry.GetBackwardDtype(operationType, inputDtype);
            }

            // Default policy: use input dtype
            return inputDtype;
        }
    }
}
