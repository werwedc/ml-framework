using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Metal compute shader for softmax activation
    /// </summary>
    public sealed class MetalSoftmaxShader : MetalComputeShaderBase
    {
        // Metal P/Invoke declarations
        [DllImport("__Internal", EntryPoint = "LoadSoftmaxShader")]
        private static extern IntPtr LoadSoftmaxShader(IntPtr device);

        public MetalSoftmaxShader(MetalBackend backend)
            : base(backend, OperatorType.Softmax, "Softmax")
        {
        }

        /// <inheritdoc/>
        protected override IntPtr InitializePipelineState()
        {
            IntPtr devicePtr = GetDevicePointer();
            return LoadSoftmaxShader(devicePtr);
        }

        /// <inheritdoc/>
        public override void Dispatch(MetalCommandBuffer commandBuffer, MetalBuffer[] inputs, MetalBuffer[] outputs, Dictionary<string, object> parameters)
        {
            if (inputs == null || inputs.Length != 1)
                throw new ArgumentException("Softmax requires exactly one input", nameof(inputs));

            if (outputs == null || outputs.Length != 1)
                throw new ArgumentException("Softmax requires exactly one output", nameof(outputs));

            // Get softmax parameters
            int axis = parameters.TryGetValue("axis", out var a) ? Convert.ToInt32(a) : -1; // Default: last axis

            // Get tensor dimensions
            int batchSize = parameters.TryGetValue("batch_size", out var bs) ? Convert.ToInt32(bs) : 1;
            int numClasses = parameters.TryGetValue("num_classes", out var nc) ? Convert.ToInt32(nc) : 1000;

            // Create command encoder
            var encoder = commandBuffer.CreateComputeCommandEncoder();

            // Set compute pipeline state
            encoder.SetComputePipelineState(PipelineState);

            // Set input and output buffers
            encoder.SetBuffer(inputs[0].NativeBuffer, 0);
            encoder.SetBuffer(outputs[0].NativeBuffer, 1);

            // Set shader parameters
            SetSoftmaxParameters(encoder, axis, batchSize, numClasses);

            // Calculate threadgroup size
            // For softmax, we use 1D threadgroups
            int threadsPerThreadgroup = 256;
            int threadgroups = (numClasses + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

            var threadsPerThreadgroupSize = new MTLSize(threadsPerThreadgroup, 1, 1);
            var threadgroupsSize = new MTLSize(threadgroups, 1, 1);

            // Dispatch
            encoder.DispatchThreadgroups(threadgroupsSize, threadsPerThreadgroupSize);
            encoder.EndEncoding();
        }

        private void SetSoftmaxParameters(MetalComputeCommandEncoder encoder,
            int axis, int batchSize, int numClasses)
        {
            int[] paramsArray = new int[] { axis, batchSize, numClasses };

            IntPtr paramsPtr = Marshal.AllocHGlobal(paramsArray.Length * sizeof(int));
            try
            {
                Marshal.Copy(paramsArray, 0, paramsPtr, paramsArray.Length);
                encoder.SetBytes(paramsPtr, paramsArray.Length * sizeof(int), 2);
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
