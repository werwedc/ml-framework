using MLFramework.MobileRuntime.Backends.Cpu.Interfaces;
using MLFramework.MobileRuntime.Backends.Cpu.Utils;
using MLFramework.MobileRuntime;

namespace MLFramework.MobileRuntime.Backends.Cpu.Executors
{
    using System;
    using System.Collections.Generic;
    using System.Runtime.CompilerServices;

    /// <summary>
    /// Executor for ReLU activation.
    /// </summary>
    public sealed class ReluExecutor : IOperatorExecutor
    {
        private readonly CpuBackend _backend;

        public OperatorType OperatorType => OperatorType.Relu;

        public ReluExecutor(CpuBackend backend)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        }

        public ITensor Execute(ITensor[] inputs, Dictionary<string, object> parameters)
        {
            if (inputs == null || inputs.Length != 1)
                throw new ArgumentException("ReLU requires exactly one input tensor.");

            var input = inputs[0];
            var outputData = input.ToArray<float>();
            var inputData = input.ToArray<float>();

            unsafe
            {
                fixed (float* inputPtr = inputData)
                fixed (float* outputPtr = outputData)
                {
                    // Copy input to output
                    for (int i = 0; i < inputData.Length; i++)
                    {
                        outputPtr[i] = inputPtr[i];
                    }

                    // Apply ReLU in-place
                    CpuVectorization.Relu(outputPtr, outputData.Length, _backend.IsVectorizationEnabled());
                }
            }

            // Create output tensor with the same shape
            // Note: In a real implementation, we'd use the tensor factory
            throw new NotImplementedException("Tensor factory integration needed");
        }

        public bool CanFuseWith(IOperatorExecutor other)
        {
            // Can be fused after Conv2D, FullyConnected
            return other is Conv2DExecutor || other is FullyConnectedExecutor;
        }

        public ITensor ExecuteFused(IOperatorExecutor[] executors, ITensor[][] inputs, Dictionary<string, object>[] parameters)
        {
            // If fused with Conv2D, apply ReLU in-place after convolution
            if (executors[0] is Conv2DExecutor conv)
            {
                return conv.ExecuteFused(new[] { this }, inputs, parameters);
            }

            return Execute(inputs[0], parameters[0]);
        }
    }
}
