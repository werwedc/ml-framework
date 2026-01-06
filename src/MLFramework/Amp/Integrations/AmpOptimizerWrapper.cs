using System;
using System.Collections.Generic;
using MLFramework.Core;
using MLFramework.Optimizers;
using RitterFramework.Core.Tensor;

namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// Wrapper for optimizers that handles AMP-specific operations
    /// </summary>
    public class AmpOptimizerWrapper : IOptimizer
    {
        private readonly IOptimizer _optimizer;
        private readonly ILossScaler _scaler;
        private readonly DataType _parameterDtype;
        private readonly DataType _gradientDtype;

        /// <summary>
        /// Gets the underlying optimizer
        /// </summary>
        public IOptimizer Optimizer => _optimizer;

        /// <summary>
        /// Gets the GradScaler
        /// </summary>
        public ILossScaler Scaler => _scaler;

        /// <summary>
        /// Gets the parameter dtype
        /// </summary>
        public DataType ParameterDtype => _parameterDtype;

        /// <summary>
        /// Gets the gradient dtype
        /// </summary>
        public DataType GradientDtype => _gradientDtype;

        /// <summary>
        /// Gets the current learning rate
        /// </summary>
        public float LearningRate => _optimizer.LearningRate;

        /// <summary>
        /// Creates a new AmpOptimizerWrapper
        /// </summary>
        /// <param name="optimizer">The underlying optimizer</param>
        /// <param name="scaler">The GradScaler</param>
        /// <param name="parameterDtype">The parameter dtype (default: BFloat16)</param>
        /// <param name="gradientDtype">The gradient dtype (default: Float32)</param>
        public AmpOptimizerWrapper(
            IOptimizer optimizer,
            ILossScaler scaler,
            DataType? parameterDtype = null,
            DataType? gradientDtype = null)
        {
            _optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
            _scaler = scaler ?? throw new ArgumentNullException(nameof(scaler));
            _parameterDtype = parameterDtype ?? DataType.BFloat16;
            _gradientDtype = gradientDtype ?? DataType.Float32;
        }

        /// <summary>
        /// Performs an optimizer step with AMP handling
        /// </summary>
        /// <param name="gradients">The gradients to use for the update</param>
        /// <returns>True if step was performed, false if skipped</returns>
        public bool Step(Dictionary<string, Tensor>? gradients = null)
        {
            return Step(gradients, checkOverflow: true, updateScale: true);
        }

        /// <summary>
        /// Performs an optimizer step with explicit gradient dictionary
        /// </summary>
        /// <param name="gradients">The gradients to use for the update</param>
        /// <param name="checkOverflow">Whether to check for overflow</param>
        /// <param name="updateScale">Whether to update the loss scale</param>
        /// <returns>True if step was performed, false if skipped</returns>
        public bool Step(
            Dictionary<string, Tensor>? gradients = null,
            bool checkOverflow = true,
            bool updateScale = true)
        {
            // Check for overflow
            if (checkOverflow && _scaler.Enabled)
            {
                bool hasOverflow = _scaler.CheckOverflow(gradients);

                // Update scale
                if (updateScale)
                {
                    _scaler.UpdateScale(hasOverflow);
                }

                if (hasOverflow)
                {
                    // Skip optimizer step on overflow
                    return false;
                }
            }

            // Unscale gradients
            var unscaledGrads = _scaler.UnscaleGradients(gradients);

            // Convert gradients to target dtype
            var convertedGrads = AmpAutogradHelper.ConvertGradientsDtype(
                unscaledGrads,
                _gradientDtype);

            // Step the optimizer
            _optimizer.Step(convertedGrads);

            return true;
        }

        /// <summary>
        /// Sets the parameters to optimize
        /// </summary>
        /// <param name="parameters">Dictionary mapping parameter names to tensors</param>
        public void SetParameters(Dictionary<string, Tensor> parameters)
        {
            _optimizer.SetParameters(parameters);
        }

        /// <summary>
        /// Performs an optimizer step with the given gradients
        /// </summary>
        /// <param name="gradients">Dictionary mapping parameter names to gradient tensors</param>
        void IOptimizer.Step(Dictionary<string, Tensor> gradients)
        {
            Step(gradients, checkOverflow: true, updateScale: true);
        }

        /// <summary>
        /// Applies a specific gradient to a specific parameter
        /// </summary>
        /// <param name="parameterName">Name of the parameter to update</param>
        /// <param name="gradient">Gradient tensor for the parameter</param>
        public void StepParameter(string parameterName, Tensor gradient)
        {
            // Unscale gradient
            var unscaledGrad = _scaler.UnscaleGradient(gradient);

            // Convert to target dtype
            var convertedGrad = AmpAutogradHelper.ConvertGradientsDtype(
                new Dictionary<string, Tensor> { { parameterName, unscaledGrad } },
                _gradientDtype
            );

            _optimizer.StepParameter(parameterName, convertedGrad[parameterName]);
        }

        /// <summary>
        /// Zeroes out all gradients
        /// </summary>
        public void ZeroGrad()
        {
            _optimizer.ZeroGrad();
        }

        /// <summary>
        /// Sets the learning rate
        /// </summary>
        /// <param name="lr">New learning rate</param>
        public void SetLearningRate(float lr)
        {
            _optimizer.SetLearningRate(lr);
        }

        /// <summary>
        /// Gets the gradients from the model
        /// </summary>
        /// <returns>Dictionary of parameter names to gradient tensors</returns>
        public Dictionary<string, Tensor> GetGradients()
        {
            // Note: IOptimizer doesn't have a GetGradients method
            throw new NotImplementedException("GetGradients is not implemented in IOptimizer");
        }

        /// <summary>
        /// Sets the gradients
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        public void SetGradients(Dictionary<string, Tensor> gradients)
        {
            // Note: IOptimizer doesn't have a SetGradients method
            throw new NotImplementedException("SetGradients is not implemented in IOptimizer");
        }

        /// <summary>
        /// Gets the parameters
        /// </summary>
        /// <returns>Dictionary of parameter names to parameter tensors</returns>
        public Dictionary<string, Tensor> GetParameters()
        {
            // Note: IOptimizer doesn't have a GetParameters method
            throw new NotImplementedException("GetParameters is not implemented in IOptimizer");
        }

        /// <summary>
        /// Loads optimizer state
        /// </summary>
        /// <param name="state">The state to load</param>
        public void LoadState(object state)
        {
            // Note: IOptimizer doesn't have a LoadState method
            throw new NotImplementedException("LoadState is not implemented in IOptimizer");
        }

        /// <summary>
        /// Gets optimizer state
        /// </summary>
        /// <returns>The optimizer state</returns>
        public object GetState()
        {
            // Note: IOptimizer doesn't have a GetState method
            throw new NotImplementedException("GetState is not implemented in IOptimizer");
        }
    }
}
