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
        private readonly int _axis;

        /// <summary>
        /// Initializes a new instance of the <see cref="VMapTransform"/> class.
        /// </summary>
        /// <param name="original">The original function to transform.</param>
        /// <param name="axis">The batch axis (default: 0).</param>
        public VMapTransform(Delegate original, int axis = 0)
            : base("vmap", TransformationType.Vectorization)
        {
            _axis = axis;
            ValidateDelegate(original);
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
            return (Tensor batchInput) =>
            {
                // Validate input has batch dimension
                if (batchInput.Dimensions <= _axis)
                {
                    throw new ArgumentException(
                        $"Input tensor must have at least {_axis + 1} dimensions for axis={_axis}. " +
                        $"Current dimensions: {batchInput.Dimensions}",
                        nameof(batchInput));
                }

                var batchSize = batchInput.Shape[_axis];
                var outputs = new List<Tensor>();

                // Iterate over batch dimension
                for (int i = 0; i < batchSize; i++)
                {
                    // Extract single element from batch
                    var singleInput = batchInput.Take(_axis, i);
                    var output = original(singleInput);
                    outputs.Add(output);
                }

                // Stack outputs along batch dimension
                return outputs.Stack(_axis);
            };
        }

        /// <summary>
        /// Creates a wrapper function for double-input delegates.
        /// </summary>
        private Func<Tensor, Tensor, Tensor> CreateDoubleInputWrapper(Func<Tensor, Tensor, Tensor> original)
        {
            return (Tensor batchInput1, Tensor batchInput2) =>
            {
                // Validate inputs have same batch size
                if (batchInput1.Dimensions <= _axis)
                {
                    throw new ArgumentException(
                        $"First input tensor must have at least {_axis + 1} dimensions for axis={_axis}. " +
                        $"Current dimensions: {batchInput1.Dimensions}",
                        nameof(batchInput1));
                }

                if (batchInput2.Dimensions <= _axis)
                {
                    throw new ArgumentException(
                        $"Second input tensor must have at least {_axis + 1} dimensions for axis={_axis}. " +
                        $"Current dimensions: {batchInput2.Dimensions}",
                        nameof(batchInput2));
                }

                var batchSize1 = batchInput1.Shape[_axis];
                var batchSize2 = batchInput2.Shape[_axis];

                if (batchSize1 != batchSize2)
                {
                    throw new ArgumentException(
                        $"Batch dimensions must match at axis {_axis}: {batchSize1} != {batchSize2}");
                }

                var batchSize = batchSize1;
                var outputs = new List<Tensor>();

                for (int i = 0; i < batchSize; i++)
                {
                    var singleInput1 = batchInput1.Take(_axis, i);
                    var singleInput2 = batchInput2.Take(_axis, i);
                    var output = original(singleInput1, singleInput2);
                    outputs.Add(output);
                }

                return outputs.Stack(_axis);
            };
        }
    }
}
