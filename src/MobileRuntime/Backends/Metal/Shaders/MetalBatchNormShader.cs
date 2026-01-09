using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Metal compute shader for batch normalization
    /// </summary>
    public sealed class MetalBatchNormShader : MetalComputeShaderBase
    {
        // Metal P/Invoke declarations
        [DllImport("__Internal", EntryPoint = "LoadBatchNormShader")]
        private static extern IntPtr LoadBatchNormShader(IntPtr device);

        public MetalBatchNormShader(MetalBackend backend)
            : base(backend, OperatorType.BatchNorm, "BatchNorm")
        {
        }

        /// <inheritdoc/>
        protected override IntPtr InitializePipelineState()
        {
            IntPtr devicePtr = GetDevicePointer();
            return LoadBatchNormShader(devicePtr);
        }

        /// <inheritdoc/>
        public override void Dispatch(MetalCommandBuffer commandBuffer, MetalBuffer[] inputs, MetalBuffer[] outputs, Dictionary<string, object> parameters)
        {
            if (inputs == null || inputs.Length == 0)
                throw new ArgumentException("BatchNorm requires at least one input", nameof(inputs));

            if (outputs == null || outputs.Length == 0)
                throw new ArgumentException("BatchNorm requires at least one output", nameof(outputs));

            // Get batch normalization parameters
            int numChannels = parameters.TryGetValue("num_channels", out var nc) ? Convert.ToInt32(nc) : 64;
            float epsilon = parameters.TryGetValue("epsilon", out var eps) ? Convert.ToSingle(eps) : 1e-5f;

            // Get tensor dimensions
            int batchSize = parameters.TryGetValue("batch_size", out var bs) ? Convert.ToInt32(bs) : 1;
            int height = parameters.TryGetValue("height", out var h) ? Convert.ToInt32(h) : 224;
            int width = parameters.TryGetValue("width", out var w) ? Convert.ToInt32(w) : 224;

            int length = batchSize * numChannels * height * width;

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
            SetBatchNormParameters(encoder, numChannels, epsilon, batchSize, height, width, length);

            // Calculate threadgroup size
            // For batch normalization, we use 1D threadgroups
            int threadsPerThreadgroup = 256;
            int threadgroups = (length + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

            var threadsPerThreadgroupSize = new MTLSize(threadsPerThreadgroup, 1, 1);
            var threadgroupsSize = new MTLSize(threadgroups, 1, 1);

            // Dispatch
            encoder.DispatchThreadgroups(threadgroupsSize, threadsPerThreadgroupSize);
            encoder.EndEncoding();
        }

        private void SetBatchNormParameters(MetalComputeCommandEncoder encoder,
            int numChannels, float epsilon, int batchSize, int height, int width, int length)
        {
            int[] intParams = new int[] { numChannels, batchSize, height, width, length };
            float[] floatParams = new float[] { epsilon };

            IntPtr intParamsPtr = Marshal.AllocHGlobal(intParams.Length * sizeof(int));
            IntPtr floatParamsPtr = Marshal.AllocHGlobal(floatParams.Length * sizeof(float));

            try
            {
                Marshal.Copy(intParams, 0, intParamsPtr, intParams.Length);
                Marshal.Copy(floatParams, 0, floatParamsPtr, floatParams.Length);

                encoder.SetBytes(intParamsPtr, intParams.Length * sizeof(int), 10);
                encoder.SetBytes(floatParamsPtr, floatParams.Length * sizeof(float), 11);
            }
            finally
            {
                Marshal.FreeHGlobal(intParamsPtr);
                Marshal.FreeHGlobal(floatParamsPtr);
            }
        }

        private IntPtr GetDevicePointer()
        {
            return IntPtr.Zero;
        }
    }
}
