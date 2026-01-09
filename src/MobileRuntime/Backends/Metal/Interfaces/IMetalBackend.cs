using System;
using System.Collections.Generic;

namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Interface for Metal-based GPU backend for iOS devices
    /// </summary>
    public interface IMetalBackend : IDisposable
    {
        /// <summary>
        /// Gets the backend name
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets the backend capabilities
        /// </summary>
        MetalBackendCapabilities Capabilities { get; }

        /// <summary>
        /// Allocates a Metal buffer of the specified size
        /// </summary>
        IMetalBuffer AllocateBuffer(long size);

        /// <summary>
        /// Frees a Metal buffer
        /// </summary>
        void FreeBuffer(IMetalBuffer buffer);

        /// <summary>
        /// Executes a single operator
        /// </summary>
        ITensor Execute(OperatorDescriptor op, ITensor[] inputs, Dictionary<string, object> parameters);

        /// <summary>
        /// Executes a batch of operators
        /// </summary>
        ITensor[] ExecuteBatch(OperatorDescriptor[] ops, Dictionary<uint, ITensor> tensorRegistry);

        /// <summary>
        /// Gets a compute shader for the specified operator type
        /// </summary>
        IMetalComputeShader GetComputeShader(OperatorType opType);

        /// <summary>
        /// Waits for all pending operations to complete
        /// </summary>
        void WaitForCompletion();

        /// <summary>
        /// Gets information about the Metal device
        /// </summary>
        MetalDeviceInfo GetDeviceInfo();
    }
}
