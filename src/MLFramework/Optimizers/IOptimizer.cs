using System.Collections.Generic;
using RitterFramework.Core.Tensor;

namespace MLFramework.Optimizers
{
    /// <summary>
    /// Base interface for all optimizers
    /// </summary>
    public interface IOptimizer
    {
        /// <summary>
        /// Current learning rate
        /// </summary>
        float LearningRate { get; }

        /// <summary>
        /// Sets the parameters to optimize
        /// </summary>
        /// <param name="parameters">Dictionary mapping parameter names to tensors</param>
        void SetParameters(Dictionary<string, Tensor> parameters);

        /// <summary>
        /// Performs an optimizer step with the given gradients
        /// </summary>
        /// <param name="gradients">Dictionary mapping parameter names to gradient tensors</param>
        void Step(Dictionary<string, Tensor> gradients);

        /// <summary>
        /// Applies a specific gradient to a specific parameter
        /// </summary>
        /// <param name="parameterName">Name of the parameter to update</param>
        /// <param name="gradient">Gradient tensor for the parameter</param>
        void StepParameter(string parameterName, Tensor gradient);

        /// <summary>
        /// Zeroes out all gradients
        /// </summary>
        void ZeroGrad();

        /// <summary>
        /// Sets the learning rate
        /// </summary>
        /// <param name="lr">New learning rate</param>
        void SetLearningRate(float lr);
    }
}
