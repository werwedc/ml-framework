using System;
using System.Collections.Generic;

namespace MLFramework.IR.Graph
{
    using MLFramework.IR.Attributes;
    using MLFramework.IR.Types;

    /// <summary>
    /// Represents a module in the High-Level IR.
    /// A module is a collection of functions and constants.
    /// </summary>
    public class HLIRModule
    {
        /// <summary>Gets the IR context for this module.</summary>
        public IRContext Context { get; }

        /// <summary>Gets the functions in this module.</summary>
        public List<HLIRFunction> Functions { get; }

        /// <summary>Gets the constants in this module (keyed by name).</summary>
        public Dictionary<string, IIRAttribute> Constants { get; }

        /// <summary>
        /// Initializes a new instance of the HLIRModule class with a new context.
        /// </summary>
        public HLIRModule()
            : this(new IRContext())
        {
        }

        /// <summary>
        /// Initializes a new instance of the HLIRModule class with the specified context.
        /// </summary>
        /// <param name="ctx">The IR context for this module.</param>
        public HLIRModule(IRContext ctx)
        {
            Context = ctx ?? throw new ArgumentNullException(nameof(ctx));
            Functions = new List<HLIRFunction>();
            Constants = new Dictionary<string, IIRAttribute>();
        }

        /// <summary>
        /// Creates a new function in this module.
        /// </summary>
        /// <param name="name">The name of the function.</param>
        /// <returns>The created function.</returns>
        public HLIRFunction CreateFunction(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentException("Function name cannot be null or whitespace.", nameof(name));
            }

            var function = new HLIRFunction(name, Context);
            Functions.Add(function);

            return function;
        }

        /// <summary>
        /// Adds a function to this module.
        /// </summary>
        /// <param name="function">The function to add.</param>
        public void AddFunction(HLIRFunction function)
        {
            if (function == null)
            {
                throw new ArgumentNullException(nameof(function));
            }

            Functions.Add(function);
        }

        /// <summary>
        /// Removes a function from this module.
        /// </summary>
        /// <param name="function">The function to remove.</param>
        /// <returns>True if the function was removed, false otherwise.</returns>
        public bool RemoveFunction(HLIRFunction function)
        {
            return Functions.Remove(function);
        }

        /// <summary>
        /// Gets a function by name.
        /// </summary>
        /// <param name="name">The name of the function.</param>
        /// <returns>The function with the specified name, or null if not found.</returns>
        public HLIRFunction GetFunction(string name)
        {
            foreach (var function in Functions)
            {
                if (function.Name == name)
                {
                    return function;
                }
            }

            return null;
        }

        /// <summary>
        /// Adds a constant to this module.
        /// </summary>
        /// <param name="name">The name of the constant.</param>
        /// <param name="value">The value of the constant.</param>
        public void AddConstant(string name, IIRAttribute value)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentException("Constant name cannot be null or whitespace.", nameof(name));
            }

            if (value == null)
            {
                throw new ArgumentNullException(nameof(value));
            }

            Constants[name] = value;
        }

        /// <summary>
        /// Gets a constant by name.
        /// </summary>
        /// <param name="name">The name of the constant.</param>
        /// <returns>The constant with the specified name, or null if not found.</returns>
        public IIRAttribute GetConstant(string name)
        {
            return Constants.TryGetValue(name, out var value) ? value : null;
        }

        /// <summary>
        /// Removes a constant from this module.
        /// </summary>
        /// <param name="name">The name of the constant to remove.</param>
        /// <returns>True if the constant was removed, false otherwise.</returns>
        public bool RemoveConstant(string name)
        {
            return Constants.Remove(name);
        }

        /// <summary>
        /// Gets the main function of this module.
        /// This is typically the function named "main" or the first function.
        /// </summary>
        /// <returns>The main function, or null if no functions exist.</returns>
        public HLIRFunction GetMainFunction()
        {
            // Try to find "main" function first
            var mainFunc = GetFunction("main");
            if (mainFunc != null)
            {
                return mainFunc;
            }

            // Otherwise return the first function
            return Functions.Count > 0 ? Functions[0] : null;
        }

        public override string ToString()
        {
            return $"Module ({Functions.Count} functions, {Constants.Count} constants)";
        }
    }
}
