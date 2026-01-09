using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Metal compute shader for fully connected (dense) layer
    /// </summary>
    public sealed class MetalFullyConnectedShader : MetalComputeShaderBase
    {
        // Metal P/Invoke declarations
        [DllImport("__Internal", EntryPoint = "LoadFullyConnectedShader")]
        private static extern IntPtr LoadFullyConnectedShader(IntPtr device);

        public MetalFullyConnectedShader(MetalBackend backend)
            : base(backend, OperatorType.FullyConnected, "FullyConnected")
        {
        }

        /// <inheritdoc/>
        protected override IntPtr InitializePipelineState()
        {
            IntPtr devicePtr = GetDevicePointer();
            return LoadFullyConnectedShader(devicePtr);
        }

        /// <inheritdoc/>
        public override void Dispatch(MetalCommandBuffer commandBuffer, MetalBuffer[] inputs, MetalBuffer[] outputs, Dictionary<string, object> parameters)
        {
            if (inputs == null || inputs.Length == 0)
                throw new ArgumentException("FullyConnected requires at least one input", nameof(inputs));

            if (outputs == null || outputs.Length == 0)
                throw new ArgumentException("FullyConnected requires at least one output", nameof(outputs));

            // Get layer parameters
            int inputSize = parameters.TryGetValue("input_size", out var isz) ? Convert.ToInt32(isz) : 1024;
            int outputSize = parameters.TryGetValue("output_size", out var osz) ? Convert.ToInt32(osz) : 1024;
            int batchSize = parameters.TryGetValue("batch_size", out var bs) ? Convert.ToInt32(bs) : 1;

            // Create command encoder
            var encoder = commandBuffer.CreateComputeCommandEncoder();

            // Set compute pipeline state
            encoder.SetComputePipelineState(PipelineState);

            // Set input and output buffers
            for (int i = 0; i < inputs.Length; i++)
            {
                encoder.SetBuffer(inputs[i].NativeBuffer, i);
            }
            for (int i = 0; i < outputs.Length; i++)
            {
                encoder.SetBuffer(outputs[i].NativeBuffer, inputs.Length + i);
            }

            // Set shader parameters
            SetFullyConnectedParameters(encoder, inputSize, outputSize, batchSize);

            // Calculate threadgroup size
            // For fully connected, we use 2D threadgroups (batch, output_size)
            int threadsPerThreadgroupX = 16;
            int threadsPerThreadgroupY = 16;

            int threadgroupsX = (outputSize + threadsPerThreadgroupX - 1) / threadsPerThreadgroupX;
            int threadgroupsY = (batchSize + threadsPerThreadgroupY - 1) / threadsPerThreadgroupY;

            var threadsPerThreadgroup = new MTLSize(threadsPerThreadgroupX, threadsPerThreadgroupY, 1);
            var threadgroups = new MTLSize(threadgroupsX, threadgroupsY, 1);

            // Dispatch
            encoder.DispatchThreadgroups(threadgroups, threadsPerThreadgroup);
            encoder.EndEncoding();
        }

        private void SetFullyConnectedParameters(MetalComputeCommandEncoder encoder,
            int inputSize, int outputSize, int batchSize)
        {
            int[] paramsArray = new int[] { inputSize, outputSize, batchSize };

            IntPtr paramsPtr = Marshal.AllocHGlobal(paramsArray.Length * sizeof(int));
            try
            {
                Marshal.Copy(paramsArray, 0, paramsPtr, paramsArray.Length);
                encoder.SetBytes(paramsPtr, paramsArray.Length * sizeof(int), 10);
            }
            finally
            {
                Marshal.FreeHGlobal(paramsPtr);
            }
        }

        private IntPtr GetDevicePointer()
        {
            return IntPtr.Zero;
        }
    }
}
