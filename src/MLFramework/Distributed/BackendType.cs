namespace MLFramework.Distributed
{
    /// <summary>
    /// Supported backend types for distributed training.
    /// </summary>
    public enum BackendType
    {
        /// <summary>
        /// NVIDIA Collective Communications Library.
        /// Optimized for GPU-to-GPU communication on NVIDIA hardware.
        /// </summary>
        NCCL,

        /// <summary>
        /// Gloo backend for CPU and multi-GPU communication.
        /// Works on both Linux and Windows.
        /// </summary>
        Gloo,

        /// <summary>
        /// Message Passing Interface (future implementation).
        /// </summary>
        MPI,

        /// <summary>
        /// AMD ROCm Communication Collectives Library (future implementation).
        /// </summary>
        RCCL
    }
}
