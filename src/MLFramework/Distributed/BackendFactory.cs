using MLFramework.Distributed.Gloo;
using MLFramework.Distributed.NCCL;

namespace MLFramework.Distributed
{
    /// <summary>
    /// Factory for creating communication backends based on the specified type.
    /// </summary>
    public static class BackendFactory
    {
        /// <summary>
        /// Creates a new instance of the specified communication backend.
        /// </summary>
        /// <param name="backendType">The type of backend to create.</param>
        /// <returns>A new instance of the requested backend.</returns>
        /// <exception cref="ArgumentException">Thrown when the backend type is not supported.</exception>
        public static ICommunicationBackend CreateBackend(BackendType backendType)
        {
            return backendType switch
            {
                BackendType.NCCL => new NCCLBackend(),
                BackendType.Gloo => new GlooBackend(),
                BackendType.MPI => throw new NotSupportedException("MPI backend is not yet implemented"),
                BackendType.RCCL => throw new NotSupportedException("RCCL backend is not yet implemented"),
                _ => throw new ArgumentException($"Unsupported backend: {backendType}")
            };
        }

        /// <summary>
        /// Checks if a backend is available on the current system.
        /// </summary>
        /// <param name="backendType">The type of backend to check.</param>
        /// <returns>True if the backend is available, false otherwise.</returns>
        public static bool IsBackendAvailable(BackendType backendType)
        {
            return backendType switch
            {
                BackendType.NCCL => NCCLBackend.CheckAvailability(),
                BackendType.Gloo => GlooBackend.CheckAvailability(),
                BackendType.MPI => false,
                BackendType.RCCL => false,
                _ => false
            };
        }
    }
}
