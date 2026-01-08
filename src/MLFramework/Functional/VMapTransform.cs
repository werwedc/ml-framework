using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.Tensor;

namespace MLFramework.Functional
{
    /// <summary>
    /// Implements the vmap (vectorization map) transformation that automatically
    /// transforms functions operating on single data points to work on batches.
    /// </summary>
    public class VMapTransform : BaseTransformation
    {
        private readonly int[] _axes;  // -1 means no vectorization for that param

        /// <summary>
        /// Initializes a new instance of the <see cref="VMapTransform"/> class with multi-axis support.
        /// </summary>
        /// <param name="original">The original function to transform.</param>
        /// <param name="in_axes">Array of axes for each parameter. Use null for non-batched parameters.</param>
        public VMapTransform(Delegate original, object[] in_axes)
            : base("vmap", TransformationType.Vectorization)
        {
            ValidateDelegate(original);

            // Normalize in_axes to int array
            var paramCount = original.Method.GetParameters().Length;
            _axes = new int[paramCount];

            if (in_axes == null)
            {
                // Default: vectorize all params on axis 0
                for (int i = 0; i < paramCount; i++)
                    _axes[i] = 0;
            }
            else
            {
                if (in_axes.Length != paramCount)
                {
                    throw new ArgumentException($"in_axes length ({in_axes.Length}) must match parameter count ({paramCount})");
                }

                for (int i = 0; i < paramCount; i++)
                {
                    if (in_axes[i] == null)
                    {
                        _axes[i] = -1;  // Don't vectorize this parameter
                    }
                    else if (in_axes[i] is int axis)
                    {
                        _axes[i] = axis;
                    }
                    else
                    {
                        throw new ArgumentException($"in_axes[{i}] must be int or null");
                    }
                }
            }

            ValidateInAxes();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="VMapTransform"/> class (legacy constructor for backward compatibility).
        /// </summary>
        /// <param name="original">The original function to transform.</param>
        /// <param name="axis">The batch axis (default: 0).</param>
        public VMapTransform(Delegate original, int axis = 0)
            : base("vmap", TransformationType.Vectorization)
        {
            ValidateDelegate(original);

            // Normalize to int array with all parameters using the same axis
            var paramCount = original.Method.GetParameters().Length;
            _axes = new int[paramCount];

            for (int i = 0; i < paramCount; i++)
                _axes[i] = axis;

            ValidateInAxes();
        }

        /// <summary>
        /// Validates the in_axes configuration.
        /// </summary>
        private void ValidateInAxes()
        {
            // Check that at least one axis is vectorized
            if (_axes.All(axis => axis == -1))
            {
                throw new ArgumentException("At least one parameter must be vectorized (axis != null)");
            }

            // Check that all vectorized axes are non-negative
            for (int i = 0; i < _axes.Length; i++)
            {
                if (_axes[i] < -1)
                {
                    throw new ArgumentException($"Invalid axis {_axes[i]} for parameter {i}");
                }
            }
        }

        /// <summary>
        /// Applies the vmap transformation to a delegate.
        /// </summary>
        /// <param name="original">The original delegate to transform.</param>
        /// <returns>A new delegate with the vmap transformation applied.</returns>
        public override Delegate Transform(Delegate original)
        {
            var method = original.Method;
            var returnType = method.ReturnType;

            if (!typeof(Tensor).IsAssignableFrom(returnType))
            {
                throw new NotSupportedException("vmap only supports functions returning Tensor");
            }

            var parameters = method.GetParameters();

            // Handle single input: Func<Tensor, Tensor>
            if (parameters.Length == 1 &&
                typeof(Tensor).IsAssignableFrom(parameters[0].ParameterType))
            {
                return CreateSingleInputWrapper((Func<Tensor, Tensor>)original);
            }

            // Handle double input: Func<Tensor, Tensor, Tensor>
            if (parameters.Length == 2 &&
                typeof(Tensor).IsAssignableFrom(parameters[0].ParameterType) &&
                typeof(Tensor).IsAssignableFrom(parameters[1].ParameterType))
            {
                return CreateDoubleInputWrapper((Func<Tensor, Tensor, Tensor>)original);
            }

            throw new NotSupportedException(
                $"Unsupported delegate signature for vmap: {parameters.Length} parameters. " +
                "Supported: Func<Tensor, Tensor> and Func<Tensor, Tensor, Tensor>");
        }

        /// <summary>
        /// Creates a wrapper function for single-input delegates.
        /// </summary>
        private Func<Tensor, Tensor> CreateSingleInputWrapper(Func<Tensor, Tensor> original)
        {
            int axis = _axes[0];

            return (Tensor input) =>
            {
                if (axis == -1)
                {
                    // No vectorization needed
                    return original(input);
                }

                // Validate input has batch dimension
                if (input.Dimensions <= axis)
                {
                    throw new ArgumentException(
                        $"Input tensor must have at least {axis + 1} dimensions for axis={axis}. " +
                        $"Current dimensions: {input.Dimensions}",
                        nameof(input));
                }

                var batchSize = input.Shape[axis];
                var outputs = new List<Tensor>();

                // Iterate over batch dimension
                for (int i = 0; i < batchSize; i++)
                {
                    // Extract single element from batch
                    var singleInput = input.Take(axis, i);
                    var output = original(singleInput);
                    outputs.Add(output);
                }

                // Stack outputs along batch dimension
                return outputs.Stack(axis);
            };
        }

        /// <summary>
        /// Creates a wrapper function for double-input delegates.
        /// </summary>
        private Func<Tensor, Tensor, Tensor> CreateDoubleInputWrapper(Func<Tensor, Tensor, Tensor> original)
        {
            int axis1 = _axes[0];
            int axis2 = _axes[1];

            return (Tensor input1, Tensor input2) =>
            {
                // Determine batch dimension
                int? batchSize = null;

                if (axis1 != -1)
                {
                    if (input1.Dimensions <= axis1)
                        throw new ArgumentException($"Input1 must have at least {axis1 + 1} dimensions for axis={axis1}. " +
                            $"Current dimensions: {input1.Dimensions}", nameof(input1));
                    batchSize = input1.Shape[axis1];
                }

                if (axis2 != -1)
                {
                    if (input2.Dimensions <= axis2)
                        throw new ArgumentException($"Input2 must have at least {axis2 + 1} dimensions for axis={axis2}. " +
                            $"Current dimensions: {input2.Dimensions}", nameof(input2));
                    int batchSize2 = input2.Shape[axis2];
                    if (batchSize.HasValue && batchSize.Value != batchSize2)
                        throw new ArgumentException($"Batch dimensions must match: {batchSize.Value} != {batchSize2}");
                    batchSize = batchSize ?? batchSize2;
                }

                // Handle case where no vectorization is needed
                if (batchSize == null)
                {
                    return original(input1, input2);
                }

                int finalBatchSize = batchSize.Value;
                var outputs = new List<Tensor>();

                for (int i = 0; i < finalBatchSize; i++)
                {
                    var singleInput1 = axis1 != -1 ? input1.Take(axis1, i) : input1;
                    var singleInput2 = axis2 != -1 ? input2.Take(axis2, i) : input2;
                    var output = original(singleInput1, singleInput2);
                    outputs.Add(output);
                }

                // Determine output axis (first non -1 axis)
                int outputAxis = axis1 != -1 ? axis1 : axis2;
                return outputs.Stack(outputAxis);
            };
        }
    }
}
