namespace MLFramework.IR.Backend.GPU
{
    /// <summary>
    /// CUDA compute capability versions
    /// </summary>
    public enum ComputeCapability
    {
        /// <summary>
        /// Volta architecture
        /// </summary>
        SM_70,

        /// <summary>
        /// Turing architecture
        /// </summary>
        SM_75,

        /// <summary>
        /// Ampere architecture
        /// </summary>
        SM_80,

        /// <summary>
        /// Hopper architecture
        /// </summary>
        SM_90
    }
}
