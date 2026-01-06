using RitterFramework.Core.Tensor;

namespace MLFramework.Modules
{
    /// <summary>
    /// Base interface for all neural network modules
    /// </summary>
    public interface IModule
    {
        /// <summary>
        /// Gets all parameters in the module
        /// </summary>
        IEnumerable<Tensor> Parameters { get; }

        /// <summary>
        /// Gets the name of the module type
        /// </summary>
        string ModuleType { get; }

        /// <summary>
        /// Sets whether the module is in training mode
        /// </summary>
        bool IsTraining { get; set; }

        /// <summary>
        /// Forward pass of the module
        /// </summary>
        /// <param name="input">Input tensor</param>
        /// <returns>Output tensor</returns>
        Tensor Forward(Tensor input);

        /// <summary>
        /// Applies a function to all parameters in the module
        /// </summary>
        /// <param name="action">Action to apply to each parameter</param>
        void ApplyToParameters(Action<Tensor> action);

        /// <summary>
        /// Sets the requires_grad flag for all parameters
        /// </summary>
        /// <param name="requiresGrad">Whether parameters require gradients</param>
        void SetRequiresGrad(bool requiresGrad);
    }
}
