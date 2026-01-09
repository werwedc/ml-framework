using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Metal compute shader for element-wise operations
    /// </summary>
    public sealed class MetalElementWiseShader : MetalComputeShaderBase
    {
        // Metal P/Invoke declarations
        [DllImport("__Internal", EntryPoint = "LoadElementWiseShader")]
        private static extern IntPtr LoadElementWiseShader(IntPtr device);

        private readonly OperatorType _opType;

        public MetalElementWiseShader(MetalBackend backend, OperatorType opType)
            : base(backend, opType, opType.ToString())
        {
            _opType = opType;
        }

        /// <inheritdoc/>
        protected override IntPtr InitializePipelineState()
        {
            IntPtr devicePtr = GetDevicePointer();
            return LoadElementWiseShader(devicePtr);
        }

        /// <inheritdoc/>
        public override void Dispatch(MetalCommandBuffer commandBuffer, MetalBuffer[] inputs, MetalBuffer[] outputs, Dictionary<string, object> parameters)
        {
            if (inputs == null || inputs.Length == 0)
                throw new ArgumentException("Element-wise operation requires at least one input", nameof(inputs));

            if (outputs == null || outputs.Length == 0)
                throw new ArgumentException("Element-wise operation requires at least one output", nameof(outputs));

            // Get tensor dimensions
            int length = parameters.TryGetValue("length", out var l) ? Convert.ToInt32(l) : (int)(inputs[0].Length / sizeof(float));

            // Validate input count based on operation type
            int requiredInputs = GetRequiredInputCount(_opType);
            if (inputs.Length < requiredInputs)
                throw new ArgumentException($"{_opType} requires at least {requiredInputs} input(s), got {inputs.Length}");

            // Create command encoder
            var encoder = commandBuffer.CreateComputeCommandEncoder();

            // Set compute pipeline state
            encoder.SetComputePipelineState(PipelineState);

            // Set input buffers
            for (int i = 0; i < inputs.Length; i++)
            {
                encoder.SetBuffer(inputs[i].NativeBuffer, i);
            }

            // Set output buffers
            for (int i = 0; i < outputs.Length; i++)
            {
                encoder.SetBuffer(outputs[i].NativeBuffer, inputs.Length + i);
            }

            // Set operation type and length parameters
            IntPtr paramsPtr = Marshal.AllocHGlobal(2 * sizeof(int));
            try
            {
                Marshal.WriteInt32(paramsPtr, (int)_opType);
                Marshal.WriteInt32(paramsPtr + sizeof(int), length);
                encoder.SetBytes(paramsPtr, 2 * sizeof(int), inputs.Length + outputs.Length);
            }
            finally
            {
                Marshal.FreeHGlobal(paramsPtr);
            }

            // Calculate threadgroup size
            // For element-wise operations, we use 1D threadgroups
            int threadsPerThreadgroup = 256;
            int threadgroups = (length + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

            var threadsPerThreadgroupSize = new MTLSize(threadsPerThreadgroup, 1, 1);
            var threadgroupsSize = new MTLSize(threadgroups, 1, 1);

            // Dispatch
            encoder.DispatchThreadgroups(threadgroupsSize, threadsPerThreadgroupSize);
            encoder.EndEncoding();
        }

        private int GetRequiredInputCount(OperatorType opType)
        {
            switch (opType)
            {
                case OperatorType.Add:
                case OperatorType.Sub:
                case OperatorType.Mul:
                case OperatorType.Div:
                    return 2; // Binary operations
                case OperatorType.Sigmoid:
                case OperatorType.Tanh:
                case OperatorType.LeakyRelu:
                    return 1; // Unary operations
                default:
                    return 1;
            }
        }

        private IntPtr GetDevicePointer()
        {
            return IntPtr.Zero;
        }
    }
}
