using MLFramework.NN;
using RitterFramework.Core.Tensor;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Layers.TensorParallel
{
    /// <summary>
    /// Extension methods for working with TP modules and parameters.
    /// </summary>
    public static class TPGradExtensions
    {
        /// <summary>
        /// Gets all trainable parameters from a module recursively.
        /// </summary>
        /// <param name="module">Module to collect parameters from</param>
        /// <returns>List of trainable parameters</returns>
        public static List<Parameter> GetTrainableParameters(this Module module)
        {
            var parameters = new List<Parameter>();
            CollectParameters(module, parameters);
            return parameters;
        }

        /// <summary>
        /// Recursively collects parameters from a module and its submodules.
        /// </summary>
        private static void CollectParameters(Module module, List<Parameter> parameters)
        {
            foreach (var param in module.GetParameters())
            {
                if (param.RequiresGrad)
                {
                    parameters.Add(param);
                }
            }

            // Note: The base Module class doesn't have a Modules property,
            // so we skip recursive collection for now.
            // If/when Module has submodules, we would iterate here:
            // foreach (var submodule in module.Modules)
            // {
            //     CollectParameters(submodule, parameters);
            // }
        }

        /// <summary>
        /// Zeros all gradients in a module.
        /// </summary>
        /// <param name="module">Module to zero gradients for</param>
        public static void ZeroGrad(this Module module)
        {
            foreach (var param in module.GetParameters())
            {
                if (param.Gradient != null)
                {
                    param.Gradient.Fill(0);
                }
            }
        }

        /// <summary>
        /// Zeros gradients for a collection of parameters.
        /// </summary>
        /// <param name="parameters">Parameters to zero gradients for</param>
        public static void ZeroGrad(this IEnumerable<Parameter> parameters)
        {
            foreach (var param in parameters)
            {
                if (param.Gradient != null)
                {
                    param.Gradient.Fill(0);
                }
            }
        }

        /// <summary>
        /// Enables gradient computation for all parameters in a module.
        /// </summary>
        /// <param name="module">Module to enable gradients for</param>
        public static void EnableGrad(this Module module)
        {
            foreach (var param in module.GetParameters())
            {
                param.RequiresGrad = true;
            }
        }

        /// <summary>
        /// Disables gradient computation for all parameters in a module.
        /// </summary>
        /// <param name="module">Module to disable gradients for</param>
        public static void DisableGrad(this Module module)
        {
            foreach (var param in module.GetParameters())
            {
                param.RequiresGrad = false;
            }
        }

        /// <summary>
        /// Gets the total number of trainable parameters in a module.
        /// </summary>
        /// <param name="module">Module to count parameters for</param>
        /// <returns>Total number of trainable parameters</returns>
        public static long GetTrainableParameterCount(this Module module)
        {
            long total = 0;
            foreach (var param in module.GetParameters())
            {
                if (param.RequiresGrad)
                {
                    total += param.Size;
                }
            }
            return total;
        }

        /// <summary>
        /// Checks if any parameter has a non-zero gradient.
        /// </summary>
        /// <param name="parameters">Parameters to check</param>
        /// <returns>True if any parameter has a non-zero gradient</returns>
        public static bool HasNonZeroGradients(this IEnumerable<Parameter> parameters)
        {
            foreach (var param in parameters)
            {
                if (param.Gradient != null)
                {
                    for (int i = 0; i < param.Gradient.Size; i++)
                    {
                        if (param.Gradient.Data[i] != 0f)
                        {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// Gets a dictionary mapping parameter names to their gradients.
        /// </summary>
        /// <param name="module">Module to get gradients from</param>
        /// <returns>Dictionary of parameter names to gradients</returns>
        public static Dictionary<string, Tensor> GetGradientsDict(this Module module)
        {
            var dict = new Dictionary<string, Tensor>();
            foreach (var (name, param) in module.GetNamedParameters())
            {
                if (param.Gradient != null)
                {
                    dict[name] = param.Gradient;
                }
            }
            return dict;
        }
    }
}
