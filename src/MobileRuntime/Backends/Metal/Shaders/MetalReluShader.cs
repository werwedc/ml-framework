using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Metal compute shader for ReLU activation
    /// </summary>
    public sealed class MetalReluShader : MetalComputeShaderBase
    {
        // Metal P/Invoke declarations
        [DllImport("__Internal", EntryPoint = "LoadReluShader")]
        private static extern IntPtr LoadReluShader(IntPtr device);

        public MetalReluShader(MetalBackend backend)
            : base(backend, OperatorType.Relu, "Relu")
        {
        }

        /// <inheritdoc/>
        protected override IntPtr InitializePipelineState()
        {
            IntPtr devicePtr = GetDevicePointer();
            return LoadReluShader(devicePtr);
        }

        /// <inheritdoc/>
        public override void Dispatch(MetalCommandBuffer commandBuffer, MetalBuffer[] inputs, MetalBuffer[] outputs, Dictionary<string, object> parameters)
        {
            if (inputs == null || inputs.Length != 1)
                throw new ArgumentException("ReLU requires exactly one input", nameof(inputs));

            if (outputs == null || outputs.Length != 1)
                throw new ArgumentException("ReLU requires exactly one output", nameof(outputs));

            // Get tensor dimensions
            int length = parameters.TryGetValue("length", out var l) ? Convert.ToInt32(l) : (int)(inputs[0].Length / sizeof(float));

            // Create command encoder
            var encoder = commandBuffer.CreateComputeCommandEncoder();

            // Set compute pipeline state
            encoder.SetComputePipelineState(PipelineState);

            // Set input and output buffers
            encoder.SetBuffer(inputs[0].NativeBuffer, 0);
            encoder.SetBuffer(outputs[0].NativeBuffer, 1);

            // Set length parameter
            IntPtr lengthPtr = Marshal.AllocHGlobal(sizeof(int));
            try
            {
                Marshal.WriteInt32(lengthPtr, length);
                encoder.SetBytes(lengthPtr, sizeof(int), 2);
            }
            finally
            {
                Marshal.FreeHGlobal(lengthPtr);
            }

            // Calculate threadgroup size
            // For element-wise operations, we use 1D threadgroups
            int threadsPerThreadgroup = 256; // Optimal for Apple GPUs
            int threadgroups = (length + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

            var threadsPerThreadgroupSize = new MTLSize(threadsPerThreadgroup, 1, 1);
            var threadgroupsSize = new MTLSize(threadgroups, 1, 1);

            // Dispatch
            encoder.DispatchThreadgroups(threadgroupsSize, threadsPerThreadgroupSize);
            encoder.EndEncoding();
        }

        private IntPtr GetDevicePointer()
        {
            return IntPtr.Zero;
        }
    }
}
