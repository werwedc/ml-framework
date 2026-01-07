namespace MLFramework.Quantization.Backends
{
    /// <summary>
    /// Flags representing quantization backend capabilities.
    /// </summary>
    [Flags]
    public enum BackendCapabilityFlags
    {
        /// <summary>
        /// No capabilities.
        /// </summary>
        None = 0,

        /// <summary>
        /// Supports Int8 matrix multiplication.
        /// </summary>
        Int8MatMul = 1 << 0,

        /// <summary>
        /// Supports Int8 2D convolution.
        /// </summary>
        Int8Conv2D = 1 << 1,

        /// <summary>
        /// Supports per-channel quantization.
        /// </summary>
        PerChannelQuantization = 1 << 2,

        /// <summary>
        /// Supports mixed precision (Int8 + FP32).
        /// </summary>
        MixedPrecision = 1 << 3,

        /// <summary>
        /// Supports dynamic quantization.
        /// </summary>
        DynamicQuantization = 1 << 4,

        /// <summary>
        /// Supports static quantization.
        /// </summary>
        StaticQuantization = 1 << 5,

        /// <summary>
        /// Supports asymmetric quantization.
        /// </summary>
        AsymmetricQuantization = 1 << 6,

        /// <summary>
        /// Supports symmetric quantization.
        /// </summary>
        SymmetricQuantization = 1 << 7
    }
}
