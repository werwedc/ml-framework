using System;

namespace MobileRuntime
{
    /// <summary>
    /// Interface for computation backend
    /// </summary>
    public interface IBackend : IDisposable
    {
        /// <summary>
        /// Gets the backend type
        /// </summary>
        BackendType Type { get; }

        /// <summary>
        /// Checks if the backend is available
        /// </summary>
        bool IsAvailable();

        /// <summary>
        /// Initializes the backend
        /// </summary>
        void Initialize();

        /// <summary>
        /// Executes a computation operation
        /// </summary>
        ITensor ExecuteOperation(string operation, ITensor[] inputs, object[] parameters);
    }
}
