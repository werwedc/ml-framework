using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;

namespace MLFramework.NN
{
    /// <summary>
    /// Base class for neural network modules.
    /// </summary>
    public abstract class Module
    {
        private readonly string _name;

        /// <summary>
        /// Gets the name of this module.
        /// </summary>
        public string Name => _name;

        /// <summary>
        /// Creates a new module with the specified name.
        /// </summary>
        protected Module(string name)
        {
            _name = name ?? throw new ArgumentNullException(nameof(name));
        }

        /// <summary>
        /// Forward pass of the module.
        /// </summary>
        /// <param name="input">Input tensor.</param>
        /// <returns>Output tensor.</returns>
        public abstract Tensor Forward(Tensor input);

        /// <summary>
        /// Gets all trainable parameters of this module.
        /// </summary>
        /// <returns>Enumerable of parameters.</returns>
        public abstract IEnumerable<Parameter> GetParameters();

        /// <summary>
        /// Gets all named parameters of this module.
        /// </summary>
        /// <returns>Enumerable of name-parameter tuples.</returns>
        public abstract IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters();

        /// <summary>
        /// Enables or disables gradient computation for all parameters.
        /// </summary>
        public virtual void SetRequiresGrad(bool requiresGrad)
        {
            foreach (var param in GetParameters())
            {
                param.RequiresGrad = requiresGrad;
            }
        }
    }
}
