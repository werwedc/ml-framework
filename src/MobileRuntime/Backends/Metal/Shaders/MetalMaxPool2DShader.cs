using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Metal compute shader for 2D max pooling
    /// </summary>
    public sealed class MetalMaxPool2DShader : MetalComputeShaderBase
    {
        // Metal P/Invoke declarations
        [DllImport("__Internal", EntryPoint = "LoadMaxPool2DShader")]
        private static extern IntPtr LoadMaxPool2DShader(IntPtr device);

        public MetalMaxPool2DShader(MetalBackend backend)
            : base(backend, OperatorType.MaxPool2D, "MaxPool2D")
        {
        }

        /// <inheritdoc/>
        protected override IntPtr InitializePipelineState()
        {
            IntPtr devicePtr = GetDevicePointer();
            return LoadMaxPool2DShader(devicePtr);
        }

        /// <inheritdoc/>
        public override void Dispatch(MetalCommandBuffer commandBuffer, MetalBuffer[] inputs, MetalBuffer[] outputs, Dictionary<string, object> parameters)
        {
            if (inputs == null || inputs.Length != 1)
                throw new ArgumentException("MaxPool2D requires exactly one input", nameof(inputs));

            if (outputs == null || outputs.Length != 1)
                throw new ArgumentException("MaxPool2D requires exactly one output", nameof(outputs));

            // Get pooling parameters
            int kernelSize = parameters.TryGetValue("kernel_size", out var ks) ? Convert.ToInt32(ks) : 2;
            int stride = parameters.TryGetValue("stride", out var s) ? Convert.ToInt32(s) : 2;
            int padding = parameters.TryGetValue("padding", out var p) ? Convert.ToInt32(p) : 0;
            int channels = parameters.TryGetValue("channels", out var c) ? Convert.ToInt32(c) : 3;

            // Get tensor dimensions
            int batchSize = parameters.TryGetValue("batch_size", out var bs) ? Convert.ToInt32(bs) : 1;
            int inputHeight = parameters.TryGetValue("input_height", out var ih) ? Convert.ToInt32(ih) : 224;
            int inputWidth = parameters.TryGetValue("input_width", out var iw) ? Convert.ToInt32(iw) : 224;

            // Calculate output dimensions
            int outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
            int outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;

            // Create command encoder
            var encoder = commandBuffer.CreateComputeCommandEncoder();

            // Set compute pipeline state
            encoder.SetComputePipelineState(PipelineState);

            // Set input and output buffers
            encoder.SetBuffer(inputs[0].NativeBuffer, 0);
            encoder.SetBuffer(outputs[0].NativeBuffer, 1);

            // Set shader parameters
            SetPoolingParameters(encoder, kernelSize, stride, padding, channels,
                batchSize, inputHeight, inputWidth, outputHeight, outputWidth);

            // Calculate threadgroup size
            // For pooling, we use 2D threadgroups
            int threadsPerThreadgroupX = 16;
            int threadsPerThreadgroupY = 16;
            int threadsPerThreadgroupZ = 1;

            int threadgroupsX = (outputWidth + threadsPerThreadgroupX - 1) / threadsPerThreadgroupX;
            int threadgroupsY = (outputHeight + threadsPerThreadgroupY - 1) / threadsPerThreadgroupY;
            int threadgroupsZ = channels;

            var threadsPerThreadgroup = new MTLSize(threadsPerThreadgroupX, threadsPerThreadgroupY, threadsPerThreadgroupZ);
            var threadgroups = new MTLSize(threadgroupsX, threadgroupsY, threadgroupsZ);

            // Dispatch
            encoder.DispatchThreadgroups(threadgroups, threadsPerThreadgroup);
            encoder.EndEncoding();
        }

        private void SetPoolingParameters(MetalComputeCommandEncoder encoder,
            int kernelSize, int stride, int padding, int channels,
            int batchSize, int inputHeight, int inputWidth, int outputHeight, int outputWidth)
        {
            // Pack parameters into a struct and set as bytes
            int[] paramsArray = new int[]
            {
                kernelSize, stride, padding, channels,
                batchSize, inputHeight, inputWidth, outputHeight, outputWidth
            };

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
