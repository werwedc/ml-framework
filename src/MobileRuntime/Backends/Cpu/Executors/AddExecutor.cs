using MLFramework.MobileRuntime.Backends.Cpu.Interfaces;
using MLFramework.MobileRuntime.Backends.Cpu.Utils;
using MLFramework.MobileRuntime;

namespace MLFramework.MobileRuntime.Backends.Cpu.Executors
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Executor for element-wise addition.
    /// </summary>
    public sealed class AddExecutor : IOperatorExecutor
    {
        private readonly CpuBackend _backend;

        public OperatorType OperatorType => OperatorType.Add;

        public AddExecutor(CpuBackend backend)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        }

        public ITensor Execute(ITensor[] inputs, Dictionary<string, object> parameters)
        {
            if (inputs == null || inputs.Length != 2)
                throw new ArgumentException("Add requires exactly two input tensors.");

            var input1 = inputs[0];
            var input2 = inputs[1];

            if (input1.DataType != DataType.Float32 || input2.DataType != DataType.Float32)
                throw new ArgumentException("Only Float32 data type is currently supported.");

            var data1 = input1.ToArray<float>();
            var data2 = input2.ToArray<float>();

            if (data1.Length != data2.Length)
                throw new ArgumentException("Input tensors must have the same size.");

            var outputData = new float[data1.Length];

            unsafe
            {
                fixed (float* outputPtr = outputData)
                fixed (float* input1Ptr = data1)
                fixed (float* input2Ptr = data2)
                {
                    CpuVectorization.Add(outputPtr, input1Ptr, input2Ptr, data1.Length, _backend.IsVectorizationEnabled());
                }
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
            throw new NotSupportedException("AddExecutor does not support operator fusion.");
        }
    }
}
