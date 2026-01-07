using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;
using MLFramework.Modules;
using MLFramework.LoRA;

namespace MLFramework.NN
{
    /// <summary>
    /// Wrapper to make non-hierarchical modules work with hierarchical operations
    /// </summary>
    public class HierarchicalWrapper : IHierarchicalModule
    {
        private readonly IModule _module;

        /// <summary>
        /// Gets the wrapped module
        /// </summary>
        public IModule Module => _module;

        /// <summary>
        /// Gets the module type
        /// </summary>
        public string ModuleType => _module.ModuleType;

        /// <summary>
        /// Gets the module name
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Gets the parent module (null if root)
        /// </summary>
        public IHierarchicalModule Parent { get; set; } = null;

        /// <summary>
        /// Creates a new wrapper around a module
        /// </summary>
        public HierarchicalWrapper(IModule module, string name)
        {
            _module = module ?? throw new ArgumentNullException(nameof(module));
            Name = name ?? throw new ArgumentNullException(nameof(name));
        }

        /// <summary>
        /// Gets all child modules (wrappers have no children)
        /// </summary>
        public IEnumerable<IHierarchicalModule> Children()
        {
            return Array.Empty<IHierarchicalModule>();
        }

        /// <summary>
        /// Replaces a child module (not applicable for wrappers)
        /// </summary>
        public void ReplaceChild(string childName, IHierarchicalModule newModule)
        {
            throw new InvalidOperationException("Wrapper modules have no children to replace");
        }

        /// <summary>
        /// Adds a child module (not applicable for wrappers)
        /// </summary>
        public void AddChild(IHierarchicalModule child)
        {
            throw new InvalidOperationException("Wrapper modules cannot have children");
        }

        /// <summary>
        /// Gets a child module (not applicable for wrappers)
        /// </summary>
        public IHierarchicalModule GetChild(string name)
        {
            return null;
        }

        /// <summary>
        /// Gets all parameters (empty for wrapper)
        /// </summary>
        public IEnumerable<Parameter> GetParameters()
        {
            yield break;
        }

        /// <summary>
        /// Gets all named parameters (empty for wrapper)
        /// </summary>
        public IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters()
        {
            yield break;
        }
    }

    /// <summary>
    /// Wrapper for Linear modules with parameter and forward pass access
    /// </summary>
    public class LinearWrapper : IHierarchicalModule
    {
        private readonly IModule _module;

        /// <summary>
        /// Gets the module type
        /// </summary>
        public string ModuleType => _module.ModuleType;

        /// <summary>
        /// Gets the module name
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Gets the parent module (null if root)
        /// </summary>
        public IHierarchicalModule Parent { get; set; } = null;

        /// <summary>
        /// Gets the wrapped module
        /// </summary>
        public IModule Module => _module;

        /// <summary>
        /// Creates a new wrapper around an IModule
        /// </summary>
        public LinearWrapper(IModule module, string name)
        {
            _module = module ?? throw new ArgumentNullException(nameof(module));
            Name = name ?? throw new ArgumentNullException(nameof(name));
        }

        /// <summary>
        /// Gets all child modules (wrappers have no children)
        /// </summary>
        public IEnumerable<IHierarchicalModule> Children()
        {
            return Array.Empty<IHierarchicalModule>();
        }

        /// <summary>
        /// Replaces a child module (not applicable for wrappers)
        /// </summary>
        public void ReplaceChild(string childName, IHierarchicalModule newModule)
        {
            throw new InvalidOperationException("Wrapper modules have no children to replace");
        }

        /// <summary>
        /// Adds a child module (not applicable for wrappers)
        /// </summary>
        public void AddChild(IHierarchicalModule child)
        {
            throw new InvalidOperationException("Wrapper modules cannot have children");
        }

        /// <summary>
        /// Gets a child module (not applicable for wrappers)
        /// </summary>
        public IHierarchicalModule GetChild(string name)
        {
            return null;
        }

        /// <summary>
        /// Gets all parameters from the wrapped module
        /// </summary>
        public IEnumerable<Parameter> GetParameters()
        {
            // If the wrapped module is a Linear, try to extract its parameters
            if (_module is Linear linear)
            {
                // Convert Linear parameters to Parameter objects
                if (linear.Weight != null)
                {
                    yield return new Parameter(linear.Weight, "weight");
                }
                if (linear.Bias != null)
                {
                    yield return new Parameter(linear.Bias, "bias");
                }
            }
            // If it's a LoraLinear, get its parameters
            else if (_module is LoraLinear loraLinear)
            {
                foreach (var param in loraLinear.TrainableParameters)
                {
                    if (param is Tensor tensor)
                    {
                        yield return new Parameter(tensor, "lora_param");
                    }
                }
            }
        }

        /// <summary>
        /// Gets all named parameters from the wrapped module
        /// </summary>
        public IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters()
        {
            if (_module is Linear linear)
            {
                if (linear.Weight != null)
                {
                    yield return ("weight", new Parameter(linear.Weight, "weight"));
                }
                if (linear.Bias != null)
                {
                    yield return ("bias", new Parameter(linear.Bias, "bias"));
                }
            }
            else if (_module is LoraLinear loraLinear)
            {
                int i = 0;
                foreach (var param in loraLinear.TrainableParameters)
                {
                    if (param is Tensor tensor)
                    {
                        yield return ($"lora_{i}", new Parameter(tensor, $"lora_{i}"));
                        i++;
                    }
                }
            }
        }

        /// <summary>
        /// Forward pass through the wrapped module
        /// </summary>
        public Tensor Forward(Tensor input)
        {
            if (_module is Linear linear)
            {
                return linear.Forward(input);
            }
            else if (_module is LoraLinear loraLinear)
            {
                return loraLinear.Forward(input);
            }

            throw new InvalidOperationException($"Cannot forward through module of type {_module.GetType().Name}");
        }
    }
}
