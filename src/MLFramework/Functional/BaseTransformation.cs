using System;
using System.Linq;
using System.Reflection;
using RitterFramework.Core.Tensor;

namespace MLFramework.Functional
{
    /// <summary>
    /// Abstract base class for all functional transformations.
    /// Provides common functionality including delegate validation and context management.
    /// </summary>
    public abstract class BaseTransformation : IFunctionalTransformation
    {
        /// <summary>
        /// Gets the name of this transformation.
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Gets the type of this transformation.
        /// </summary>
        public TransformationType Type { get; }

        /// <summary>
        /// Gets the transformation context.
        /// </summary>
        protected TransformationContext Context { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="BaseTransformation"/> class.
        /// </summary>
        /// <param name="name">The name of this transformation.</param>
        /// <param name="type">The type of this transformation.</param>
        /// <param name="context">The transformation context (optional).</param>
        protected BaseTransformation(string name, TransformationType type, TransformationContext context = null)
        {
            Name = name;
            Type = type;
            Context = context ?? new TransformationContext();
        }

        /// <summary>
        /// Applies the transformation to a delegate.
        /// Subclasses must implement this method to provide specific transformation logic.
        /// </summary>
        /// <param name="original">The original delegate to transform.</param>
        /// <returns>A new delegate with the transformation applied.</returns>
        public abstract Delegate Transform(Delegate original);

        /// <summary>
        /// Validates that a delegate is suitable for transformation.
        /// </summary>
        /// <param name="del">The delegate to validate.</param>
        /// <exception cref="ArgumentNullException">Thrown when the delegate is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the delegate does not meet requirements.</exception>
        protected void ValidateDelegate(Delegate del)
        {
            if (del == null)
                throw new ArgumentNullException(nameof(del));

            // Check if delegate has TensorFunction attribute or accepts Tensor parameters
            var method = del.Method;
            var parameters = method.GetParameters();

            // Check if any parameter is a Tensor type
            bool hasTensorParameter = parameters.Any(p => typeof(Tensor).IsAssignableFrom(p.ParameterType));

            if (!hasTensorParameter)
            {
                throw new ArgumentException(
                    $"Function '{method.Name}' must accept at least one Tensor parameter for transformation.",
                    nameof(del));
            }
        }

        /// <summary>
        /// Checks if the delegate is marked as pure by the TensorFunction attribute.
        /// </summary>
        /// <param name="del">The delegate to check.</param>
        /// <returns>True if the delegate is marked as pure, false otherwise.</returns>
        protected bool IsFunctionPure(Delegate del)
        {
            var attribute = del.Method.GetCustomAttribute<TensorFunctionAttribute>();
            return attribute?.IsPure ?? true; // Default to true for safety
        }

        /// <summary>
        /// Gets the TensorFunction attribute from a delegate, if present.
        /// </summary>
        /// <param name="del">The delegate to get the attribute from.</param>
        /// <returns>The TensorFunction attribute, or null if not present.</returns>
        protected TensorFunctionAttribute GetTensorFunctionAttribute(Delegate del)
        {
            return del.Method.GetCustomAttribute<TensorFunctionAttribute>();
        }

        /// <summary>
        /// Logs a debug message if debug mode is enabled in the context.
        /// </summary>
        /// <param name="message">The message to log.</param>
        protected void LogDebug(string message)
        {
            if (Context.DebugMode)
            {
                // In a real implementation, this would use a proper logging framework
                System.Diagnostics.Debug.WriteLine($"[{Name}] {message}");
            }
        }
    }
}
