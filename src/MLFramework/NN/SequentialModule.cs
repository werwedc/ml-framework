using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.Tensor;
using MLFramework.NN;

namespace MLFramework.NN
{
    /// <summary>
    /// A module that executes a sequence of child modules in order.
    /// Used for creating stage modules from partitioned layers.
    /// </summary>
    public class SequentialModule : Module
    {
        private readonly List<Module> _modules;

        /// <summary>
        /// Number of child modules in this sequential module
        /// </summary>
        public int Count => _modules.Count;

        /// <summary>
        /// Creates a new sequential module
        /// </summary>
        /// <param name="name">Name of the module</param>
        public SequentialModule(string name = "Sequential") : base(name)
        {
            _modules = new List<Module>();
        }

        /// <summary>
        /// Creates a new sequential module with the given modules
        /// </summary>
        /// <param name="modules">List of modules to add</param>
        /// <param name="name">Name of the module</param>
        public SequentialModule(IEnumerable<Module> modules, string name = "Sequential") : base(name)
        {
            _modules = new List<Module>(modules ?? throw new ArgumentNullException(nameof(modules)));
        }

        /// <summary>
        /// Adds a module to the end of the sequence
        /// </summary>
        public void Add(Module module)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            _modules.Add(module);
        }

        /// <summary>
        /// Adds a module to the sequence and returns this module for chaining
        /// </summary>
        public SequentialModule AddModule(Module module)
        {
            Add(module);
            return this;
        }

        /// <summary>
        /// Inserts a module at the specified index
        /// </summary>
        public void Insert(int index, Module module)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            if (index < 0 || index > _modules.Count)
                throw new ArgumentOutOfRangeException(nameof(index));

            _modules.Insert(index, module);
        }

        /// <summary>
        /// Gets a module at the specified index
        /// </summary>
        public Module GetModule(int index)
        {
            if (index < 0 || index >= _modules.Count)
                throw new ArgumentOutOfRangeException(nameof(index));

            return _modules[index];
        }

        /// <summary>
        /// Gets all modules in the sequence
        /// </summary>
        public IReadOnlyList<Module> GetModules()
        {
            return _modules.AsReadOnly();
        }

        /// <summary>
        /// Executes all modules in sequence
        /// </summary>
        /// <param name="input">Input tensor</param>
        /// <returns>Output tensor from the last module</returns>
        public override Tensor Forward(Tensor input)
        {
            if (_modules.Count == 0)
                return input;

            Tensor current = input;
            foreach (var module in _modules)
            {
                current = module.Forward(current);
            }

            return current;
        }

        /// <summary>
        /// Gets all trainable parameters from all child modules
        /// </summary>
        public override IEnumerable<Parameter> GetParameters()
        {
            foreach (var module in _modules)
            {
                foreach (var param in module.GetParameters())
                {
                    yield return param;
                }
            }
        }

        /// <summary>
        /// Gets all named parameters from all child modules
        /// </summary>
        public override IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters()
        {
            foreach (var module in _modules)
            {
                foreach (var (name, param) in module.GetNamedParameters())
                {
                    yield return ($"{Name}.{name}", param);
                }
            }
        }

        /// <summary>
        /// Removes a module at the specified index
        /// </summary>
        public void RemoveAt(int index)
        {
            if (index < 0 || index >= _modules.Count)
                throw new ArgumentOutOfRangeException(nameof(index));

            _modules.RemoveAt(index);
        }

        /// <summary>
        /// Removes all modules from the sequence
        /// </summary>
        public void Clear()
        {
            _modules.Clear();
        }
    }
}
