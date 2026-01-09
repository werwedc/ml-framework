using MLFramework.MobileRuntime.Backends.Cpu.Interfaces;
using MLFramework.MobileRuntime;

namespace MLFramework.MobileRuntime.Backends.Cpu.Executors
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Executor for tensor reshape operations.
    /// </summary>
    public sealed class ReshapeExecutor : IOperatorExecutor
    {
        public OperatorType OperatorType => OperatorType.Reshape;

        public ITensor Execute(ITensor[] inputs, Dictionary<string, object> parameters)
        {
            if (inputs == null || inputs.Length != 1)
                throw new ArgumentException("Reshape requires exactly one input tensor.");

            var input = inputs[0];
            var inputData = input.ToArray<float>();
            var inputShape = input.Shape;

            // Parse target shape
            if (!parameters.TryGetValue("shape", out var shapeObj))
                throw new ArgumentException("Reshape requires 'shape' parameter.");

            var targetShape = (int[])shapeObj;

            // Handle -1 in target shape (inferred dimension)
            int inferredDimIndex = -1;
            int knownSize = 1;

            for (int i = 0; i < targetShape.Length; i++)
            {
                if (targetShape[i] == -1)
                {
                    if (inferredDimIndex != -1)
                        throw new ArgumentException("Only one dimension can be -1 in target shape.");
                    inferredDimIndex = i;
                }
                else
                {
                    knownSize *= targetShape[i];
                }
            }

            // Calculate total size
            int totalSize = 1;
            foreach (int dim in inputShape)
            {
                totalSize *= dim;
            }

            // Infer the -1 dimension
            var outputShape = (int[])targetShape.Clone();
            if (inferredDimIndex != -1)
            {
                if (totalSize % knownSize != 0)
                    throw new ArgumentException($"Cannot reshape tensor of size {totalSize} to shape with inferred dimension.");

                outputShape[inferredDimIndex] = totalSize / knownSize;
            }

            // Verify output size matches input size
            int outputSize = 1;
            foreach (int dim in outputShape)
            {
                outputSize *= dim;
            }

            if (outputSize != totalSize)
                throw new ArgumentException($"Cannot reshape tensor of size {totalSize} to shape {string.Join("x", outputShape)} (size {outputSize}).");

            // Reshape is just a view operation - data stays the same, only shape changes
            // In a real implementation, we'd create a tensor with the new shape pointing to the same data

            throw new NotImplementedException("Tensor factory integration needed");
        }

        public bool CanFuseWith(IOperatorExecutor other)
        {
            return false;
        }

        public ITensor ExecuteFused(IOperatorExecutor[] executors, ITensor[][] inputs, Dictionary<string, object>[] parameters)
        {
            throw new NotSupportedException("ReshapeExecutor does not support operator fusion.");
        }
    }
}
