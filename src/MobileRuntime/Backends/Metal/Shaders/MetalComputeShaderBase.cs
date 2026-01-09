using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Base class for Metal compute shaders
    /// </summary>
    public abstract class MetalComputeShaderBase : IMetalComputeShader
    {
        protected readonly MetalBackend Backend;
        protected readonly IntPtr PipelineState;

        // Metal P/Invoke declarations
        [DllImport("__Internal", EntryPoint = "MTLComputePipelineState_release")]
        private static extern void MTLComputePipelineState_release(IntPtr pipelineState);

        protected MetalComputeShaderBase(MetalBackend backend, OperatorType opType, string shaderName)
        {
            Backend = backend;
            OperatorTypeValue = opType;
            NameValue = shaderName;
            PipelineState = InitializePipelineState();
        }

        protected OperatorType OperatorTypeValue { get; }
        protected string NameValue { get; }

        /// <inheritdoc/>
        public string Name => NameValue;

        /// <inheritdoc/>
        public OperatorType OperatorType => OperatorTypeValue;

        /// <summary>
        /// Initializes the Metal compute pipeline state
        /// </summary>
        protected abstract IntPtr InitializePipelineState();

        /// <inheritdoc/>
        public abstract void Dispatch(MetalCommandBuffer commandBuffer, MetalBuffer[] inputs, MetalBuffer[] outputs, Dictionary<string, object> parameters);

        /// <inheritdoc/>
        public void SetBytes(IntPtr buffer, long offset, IntPtr data, long size)
        {
            // Implementation for setting bytes in the shader's argument buffer
            throw new NotImplementedException("SetBytes is not implemented for this shader");
        }

        /// <inheritdoc/>
        public void SetBuffer(MetalBuffer buffer, int index)
        {
            // Implementation for setting a buffer argument for the shader
            throw new NotImplementedException("SetBuffer is not implemented for this shader");
        }

        /// <summary>
        /// Disposes of the shader
        /// </summary>
        public void Dispose()
        {
            if (PipelineState != IntPtr.Zero)
            {
                MTLComputePipelineState_release(PipelineState);
            }
        }
    }
}
