using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using System;

namespace MLFramework.NN
{
    /// <summary>
    /// Represents a trainable parameter in a neural network.
    /// </summary>
    public class Parameter : Tensor
    {
        private readonly string _name;

        /// <summary>
        /// Gets the name of this parameter.
        /// </summary>
        public string Name => _name;

        /// <summary>
        /// Creates a new parameter with the specified data, shape, and name.
        /// </summary>
        public Parameter(float[] data, int[] shape, string name, bool requiresGrad = true, DataType dtype = DataType.Float32)
            : base(data, shape, requiresGrad, dtype)
        {
            _name = name ?? throw new ArgumentNullException(nameof(name));
        }

        /// <summary>
        /// Creates a new parameter from an existing tensor.
        /// </summary>
        public Parameter(Tensor tensor, string name, bool requiresGrad = true)
            : base(tensor.Data, tensor.Shape, requiresGrad, tensor.Dtype)
        {
            _name = name ?? throw new ArgumentNullException(nameof(name));
        }

        /// <summary>
        /// Registers a gradient hook that will be called when a gradient is computed.
        /// </summary>
        /// <param name="hook">The hook function to call with the gradient.</param>
        public void RegisterGradHook(Func<Tensor, Tensor> hook)
        {
            // Store hook to be called during backward pass
            // This will be invoked when Gradient property is set during backward
            // For simplicity in this implementation, we'll store the hook in a dictionary
            GradientHookRegistry.RegisterHook(this, hook);
        }
    }

    /// <summary>
    /// Internal registry for gradient hooks.
    /// </summary>
    internal static class GradientHookRegistry
    {
        private static readonly Dictionary<Parameter, Func<Tensor, Tensor>> _hooks = new();

        public static void RegisterHook(Parameter parameter, Func<Tensor, Tensor> hook)
        {
            _hooks[parameter] = hook ?? throw new ArgumentNullException(nameof(hook));
        }

        public static bool TryGetHook(Parameter parameter, out Func<Tensor, Tensor> hook)
        {
            return _hooks.TryGetValue(parameter, out hook);
        }

        public static void RemoveHook(Parameter parameter)
        {
            _hooks.Remove(parameter);
        }

        public static void ClearAll()
        {
            _hooks.Clear();
        }
    }
}
