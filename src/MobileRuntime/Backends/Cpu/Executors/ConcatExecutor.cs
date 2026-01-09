using MLFramework.MobileRuntime.Backends.Cpu.Interfaces;
using MLFramework.MobileRuntime;

namespace MLFramework.MobileRuntime.Backends.Cpu.Executors
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Executor for tensor concatenation.
    /// </summary>
    public sealed class ConcatExecutor : IOperatorExecutor
    {
        public OperatorType OperatorType => OperatorType.Concat;

        public ITensor Execute(ITensor[] inputs, Dictionary<string, object> parameters)
        {
            if (inputs == null || inputs.Length == 0)
                throw new ArgumentException("Concat requires at least one input tensor.");

            // Parse dimension parameter
            int dimension = 0;
            if (parameters.TryGetValue("axis", out var axisObj) || parameters.TryGetValue("dim", out axisObj))
            {
                dimension = (int)axisObj;
            }

            // Verify all inputs have the same shape except on the concatenation dimension
            var firstShape = inputs[0].Shape;
            if (dimension < 0 || dimension >= firstShape.Length)
                throw new ArgumentException($"Invalid dimension {dimension} for tensor with rank {firstShape.Length}.");

            foreach (var input in inputs)
            {
                var shape = input.Shape;
                if (shape.Length != firstShape.Length)
                    throw new ArgumentException("All input tensors must have the same rank.");

                for (int i = 0; i < shape.Length; i++)
                {
                    if (i != dimension && shape[i] != firstShape[i])
                        throw new ArgumentException($"All input tensors must have the same size on non-concatenation dimensions. Dimension {i} differs.");
                }
            }

            // Calculate output shape
            int totalDimSize = inputs.Sum(input => input.Shape[dimension]);
            var outputShape = (int[])firstShape.Clone();
            outputShape[dimension] = totalDimSize;

            // Calculate total output size
            int outputSize = 1;
            for (int i = 0; i < outputShape.Length; i++)
            {
                outputSize *= outputShape[i];
            }

            var outputData = new float[outputSize];

            // Copy data from each input tensor
            int outputOffset = 0;
            foreach (var input in inputs)
            {
                var inputData = input.ToArray<float>();

                // Calculate the size of each "slice" along the concatenation dimension
                int sliceSize = 1;
                for (int i = dimension + 1; i < input.Shape.Length; i++)
                {
                    sliceSize *= input.Shape[i];
                }

                // Copy input data to output
                int dimSize = input.Shape[dimension];
                int inputOffset = 0;

                for (int i = 0; i < dimSize; i++)
                {
                    for (int j = 0; j < sliceSize; j++)
                    {
                        outputData[outputOffset + i * sliceSize + j] = inputData[inputOffset + i * sliceSize + j];
                    }
                }

                outputOffset += dimSize * sliceSize;
            }

            // Create output tensor
            throw new NotImplementedException("Tensor factory integration needed");
        }

        public bool CanFuseWith(IOperatorExecutor other)
        {
            return false;
        }

        public ITensor ExecuteFused(IOperatorExecutor[] executors, ITensor[][] inputs, Dictionary<string, object>[] parameters)
        {
            throw new NotSupportedException("ConcatExecutor does not support operator fusion.");
        }
    }
}
