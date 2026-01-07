namespace MLFramework.Quantization.DataStructures
{
    /// <summary>
    /// Quantization mode enumeration.
    /// </summary>
    public enum QuantizationMode
    {
        /// <summary>
        /// Per-tensor symmetric quantization: single scale, zero-point = 0.
        /// </summary>
        PerTensorSymmetric,

        /// <summary>
        /// Per-tensor asymmetric quantization: single scale, single zero-point.
        /// </summary>
        PerTensorAsymmetric,

        /// <summary>
        /// Per-channel symmetric quantization: per-channel scale, zero-point = 0.
        /// </summary>
        PerChannelSymmetric,

        /// <summary>
        /// Per-channel asymmetric quantization: per-channel scale, per-channel zero-point.
        /// </summary>
        PerChannelAsymmetric
    }
}
