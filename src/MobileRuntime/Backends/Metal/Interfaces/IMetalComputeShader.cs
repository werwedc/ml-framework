using System;
using System.Collections.Generic;

namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Interface for Metal compute shader
    /// </summary>
    public interface IMetalComputeShader
    {
        /// <summary>
        /// Gets the shader name
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets the operator type this shader handles
        /// </summary>
        OperatorType OperatorType { get; }

        /// <summary>
        /// Dispatches the shader with the given inputs, outputs, and parameters
        /// </summary>
        void Dispatch(MetalCommandBuffer commandBuffer, MetalBuffer[] inputs, MetalBuffer[] outputs, Dictionary<string, object> parameters);

        /// <summary>
        /// Sets bytes in the shader's argument buffer
        /// </summary>
        void SetBytes(IntPtr buffer, long offset, IntPtr data, long size);

        /// <summary>
        /// Sets a buffer argument for the shader
        /// </summary>
        void SetBuffer(MetalBuffer buffer, int index);
    }
}
