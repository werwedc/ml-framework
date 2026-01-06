namespace MLFramework.Amp
{
    /// <summary>
    /// Precision policy for operations in AMP
    /// </summary>
    public enum OpPrecision
    {
        /// <summary>
        /// Use higher precision (FP32)
        /// </summary>
        Higher = 0,

        /// <summary>
        /// Use lower precision (FP16/BF16 based on config)
        /// </summary>
        Lower = 1,

        /// <summary>
        /// Keep original precision
        /// </summary>
        Keep = 2,

        /// <summary>
        /// Custom precision specified separately
        /// </summary>
        Custom = 3
    }
}
