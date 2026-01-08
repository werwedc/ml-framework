using System;

namespace MLFramework.Functional
{
    /// <summary>
    /// Attribute that marks methods and delegates as tensor functions with metadata.
    /// Used by transformations to understand function characteristics for optimization.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Delegate)]
    public class TensorFunctionAttribute : Attribute
    {
        /// <summary>
        /// Gets or sets whether the function is pure (no side effects).
        /// Pure functions can be safely transformed and optimized by the framework.
        /// </summary>
        public bool IsPure { get; set; } = true;

        /// <summary>
        /// Gets or sets the array of input shape specifications.
        /// Shapes can be symbolic (e.g., "B*H*W", "N*") and help transformations understand data flow.
        /// </summary>
        public string[] InputShapes { get; set; }

        /// <summary>
        /// Gets or sets the output shape specification.
        /// Helps transformations understand the expected output tensor shape.
        /// </summary>
        public string OutputShape { get; set; }

        /// <summary>
        /// Gets or sets a description of the function's purpose.
        /// Useful for debugging and documentation generation.
        /// </summary>
        public string Description { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="TensorFunctionAttribute"/> class.
        /// </summary>
        public TensorFunctionAttribute()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TensorFunctionAttribute"/> class.
        /// </summary>
        /// <param name="inputShapes">The array of input shape specifications.</param>
        /// <param name="outputShape">The output shape specification.</param>
        public TensorFunctionAttribute(string[] inputShapes, string outputShape)
        {
            InputShapes = inputShapes;
            OutputShape = outputShape;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TensorFunctionAttribute"/> class.
        /// </summary>
        /// <param name="isPure">Whether the function is pure.</param>
        /// <param name="inputShapes">The array of input shape specifications.</param>
        /// <param name="outputShape">The output shape specification.</param>
        public TensorFunctionAttribute(bool isPure, string[] inputShapes, string outputShape)
        {
            IsPure = isPure;
            InputShapes = inputShapes;
            OutputShape = outputShape;
        }
    }
}
