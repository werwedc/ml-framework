namespace MLFramework.Amp
{
    /// <summary>
    /// Kernel data type for GPU operations
    /// </summary>
    public enum KernelDtype
    {
        /// <summary>
        /// Float32 (default precision)
        /// </summary>
        Float32 = 0,

        /// <summary>
        /// Float16 (half precision)
        /// </summary>
        Float16 = 1,

        /// <summary>
        /// BFloat16 (brain float)
        /// </summary>
        BFloat16 = 2,

        /// <summary>
        /// Mixed precision (multiple dtypes)
        /// </summary>
        Mixed = 3,

        /// <summary>
        /// Automatic selection based on tensor dtype
        /// </summary>
        Auto = 4
    }
}
