namespace MLFramework.Quantization.DataStructures
{
    /// <summary>
    /// Calibration method enumeration.
    /// </summary>
    public enum CalibrationMethod
    {
        /// <summary>
        /// Min-max calibration.
        /// </summary>
        MinMax,

        /// <summary>
        /// Entropy-based calibration.
        /// </summary>
        Entropy,

        /// <summary>
        /// Percentile-based calibration.
        /// </summary>
        Percentile,

        /// <summary>
        /// Moving average calibration.
        /// </summary>
        MovingAverage
    }
}
