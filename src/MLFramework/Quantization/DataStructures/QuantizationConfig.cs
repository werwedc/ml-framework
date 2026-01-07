namespace MLFramework.Quantization.DataStructures
{
    /// <summary>
    /// Quantization configuration.
    /// </summary>
    public class QuantizationConfig
    {
        /// <summary>
        /// Gets or sets the mode for weight quantization.
        /// </summary>
        public QuantizationMode WeightQuantization { get; set; } = QuantizationMode.PerTensorSymmetric;

        /// <summary>
        /// Gets or sets the mode for activation quantization.
        /// </summary>
        public QuantizationMode ActivationQuantization { get; set; } = QuantizationMode.PerTensorSymmetric;

        /// <summary>
        /// Gets or sets the calibration strategy.
        /// </summary>
        public CalibrationMethod CalibrationMethod { get; set; } = CalibrationMethod.MinMax;

        /// <summary>
        /// Gets or sets the batch size for calibration runs.
        /// </summary>
        public int CalibrationBatchSize { get; set; } = 32;

        /// <summary>
        /// Gets or sets the bit-width and signed/unsigned type.
        /// </summary>
        public QuantizationType QuantizationType { get; set; } = QuantizationType.Int8;

        /// <summary>
        /// Gets or sets whether to fallback to FP32 for sensitive layers.
        /// </summary>
        public bool FallbackToFP32 { get; set; } = true;

        /// <summary>
        /// Gets or sets the target accuracy loss threshold (0.0 to 1.0).
        /// Layers exceeding this threshold will fallback to FP32.
        /// </summary>
        public float AccuracyThreshold { get; set; } = 0.01f;

        /// <summary>
        /// Gets or sets whether to enable per-channel quantization.
        /// </summary>
        public bool EnablePerChannelQuantization { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to enable mixed precision (Int8 + FP32).
        /// </summary>
        public bool EnableMixedPrecision { get; set; } = false;

        /// <summary>
        /// Validates the configuration.
        /// </summary>
        /// <returns>True if valid, false otherwise.</returns>
        /// <exception cref="InvalidOperationException">Thrown when configuration is invalid.</exception>
        public bool Validate()
        {
            var errors = new List<string>();

            if (CalibrationBatchSize <= 0)
            {
                errors.Add("CalibrationBatchSize must be greater than 0.");
            }

            if (AccuracyThreshold < 0.0f || AccuracyThreshold > 1.0f)
            {
                errors.Add("AccuracyThreshold must be between 0.0 and 1.0.");
            }

            if (EnablePerChannelQuantization &&
                (WeightQuantization == QuantizationMode.PerTensorSymmetric ||
                 WeightQuantization == QuantizationMode.PerTensorAsymmetric) &&
                (ActivationQuantization == QuantizationMode.PerTensorSymmetric ||
                 ActivationQuantization == QuantizationMode.PerTensorAsymmetric))
            {
                errors.Add("EnablePerChannelQuantization is true but both weight and activation modes are per-tensor.");
            }

            if (errors.Count > 0)
            {
                throw new InvalidOperationException($"Invalid QuantizationConfig: {string.Join(" ", errors)}");
            }

            return true;
        }

        /// <summary>
        /// Creates a default quantization configuration.
        /// </summary>
        /// <returns>A default configuration.</returns>
        public static QuantizationConfig CreateDefault()
        {
            return new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                CalibrationMethod = CalibrationMethod.MinMax,
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = true,
                AccuracyThreshold = 0.01f,
                EnablePerChannelQuantization = false,
                EnableMixedPrecision = false
            };
        }

        /// <summary>
        /// Creates a configuration for aggressive quantization (smaller model, potentially lower accuracy).
        /// </summary>
        /// <returns>An aggressive quantization configuration.</returns>
        public static QuantizationConfig CreateAggressive()
        {
            return new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerChannelSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                CalibrationMethod = CalibrationMethod.Entropy,
                CalibrationBatchSize = 64,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = false,
                AccuracyThreshold = 0.05f,
                EnablePerChannelQuantization = true,
                EnableMixedPrecision = false
            };
        }

        /// <summary>
        /// Creates a configuration for conservative quantization (higher accuracy, larger model).
        /// </summary>
        /// <returns>A conservative quantization configuration.</returns>
        public static QuantizationConfig CreateConservative()
        {
            return new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                CalibrationMethod = CalibrationMethod.MinMax,
                CalibrationBatchSize = 16,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = true,
                AccuracyThreshold = 0.005f,
                EnablePerChannelQuantization = false,
                EnableMixedPrecision = true
            };
        }

        /// <summary>
        /// Returns a string representation of the configuration.
        /// </summary>
        public override string ToString()
        {
            return $"QuantizationConfig(Weight: {WeightQuantization}, Activation: {ActivationQuantization}, " +
                   $"Calibration: {CalibrationMethod}, BatchSize: {CalibrationBatchSize}, " +
                   $"Type: {QuantizationType}, FallbackToFP32: {FallbackToFP32}, " +
                   $"AccuracyThreshold: {AccuracyThreshold:P1}, PerChannel: {EnablePerChannelQuantization}, " +
                   $"MixedPrecision: {EnableMixedPrecision})";
        }
    }
}
