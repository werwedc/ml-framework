using System;
using System.Threading.Tasks;

namespace MLFramework.MobileRuntime
{
    /// <summary>
    /// Core interface for mobile runtime operations.
    /// </summary>
    public interface IMobileRuntime
    {
        /// <summary>
        /// Loads a model from a file path.
        /// </summary>
        /// <param name="modelPath">Path to the model file.</param>
        /// <returns>Loaded model instance.</returns>
        IModel LoadModel(string modelPath);

        /// <summary>
        /// Loads a model from a byte array.
        /// </summary>
        /// <param name="modelBytes">Model data as bytes.</param>
        /// <returns>Loaded model instance.</returns>
        IModel LoadModel(byte[] modelBytes);

        /// <summary>
        /// Sets the maximum memory limit for the runtime.
        /// </summary>
        /// <param name="maxBytes">Maximum memory in bytes.</param>
        void SetMemoryLimit(long maxBytes);

        /// <summary>
        /// Sets the hardware backend to use for inference.
        /// </summary>
        /// <param name="backend">The backend type.</param>
        void SetHardwareBackend(BackendType backend);

        /// <summary>
        /// Gets the currently active backend.
        /// </summary>
        BackendType CurrentBackend { get; }

        /// <summary>
        /// Gets the current memory limit.
        /// </summary>
        long MemoryLimit { get; }

        /// <summary>
        /// Gets runtime information.
        /// </summary>
        /// <returns>Runtime information object.</returns>
        RuntimeInfo GetRuntimeInfo();
    }
}
