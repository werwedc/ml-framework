using System;

namespace MLFramework.MobileRuntime
{
    /// <summary>
    /// Abstract base class for mobile runtime implementations.
    /// </summary>
    public abstract class MobileRuntime : IMobileRuntime
    {
        /// <summary>
        /// Current memory limit in bytes.
        /// </summary>
        protected long _memoryLimit;

        /// <summary>
        /// Currently selected hardware backend.
        /// </summary>
        protected BackendType _currentBackend;

        /// <summary>
        /// Initializes a new instance of the MobileRuntime class.
        /// </summary>
        protected MobileRuntime()
        {
            _currentBackend = BackendType.Auto;
            _memoryLimit = long.MaxValue; // Default to unlimited
        }

        /// <summary>
        /// Gets the currently active backend.
        /// </summary>
        public virtual BackendType CurrentBackend => _currentBackend;

        /// <summary>
        /// Gets the current memory limit.
        /// </summary>
        public virtual long MemoryLimit => _memoryLimit;

        /// <summary>
        /// Sets the maximum memory limit for the runtime.
        /// </summary>
        /// <param name="maxBytes">Maximum memory in bytes.</param>
        /// <exception cref="ArgumentException">Thrown if maxBytes is not positive.</exception>
        public virtual void SetMemoryLimit(long maxBytes)
        {
            if (maxBytes <= 0)
                throw new ArgumentException("Memory limit must be positive", nameof(maxBytes));
            _memoryLimit = maxBytes;
        }

        /// <summary>
        /// Sets the hardware backend to use for inference.
        /// </summary>
        /// <param name="backend">The backend type.</param>
        public virtual void SetHardwareBackend(BackendType backend)
        {
            _currentBackend = backend;
        }

        /// <summary>
        /// Loads a model from a file path.
        /// </summary>
        /// <param name="modelPath">Path to the model file.</param>
        /// <returns>Loaded model instance.</returns>
        public abstract IModel LoadModel(string modelPath);

        /// <summary>
        /// Loads a model from a byte array.
        /// </summary>
        /// <param name="modelBytes">Model data as bytes.</param>
        /// <returns>Loaded model instance.</returns>
        public abstract IModel LoadModel(byte[] modelBytes);

        /// <summary>
        /// Gets runtime information.
        /// </summary>
        /// <returns>Runtime information object.</returns>
        public abstract RuntimeInfo GetRuntimeInfo();
    }
}
