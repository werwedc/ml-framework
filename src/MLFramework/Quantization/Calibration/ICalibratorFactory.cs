using MLFramework.Quantization.DataStructures;

namespace MLFramework.Quantization.Calibration
{
    /// <summary>
    /// Interface for creating calibrator instances based on calibration method.
    /// </summary>
    public interface ICalibratorFactory
    {
        /// <summary>
        /// Creates a calibrator instance based on the specified calibration method.
        /// </summary>
        /// <param name="method">The calibration method to use.</param>
        /// <returns>An instance of ICalibrator appropriate for the specified method.</returns>
        ICalibrator Create(CalibrationMethod method);
    }
}
