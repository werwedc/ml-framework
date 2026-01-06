using MLFramework.Core;

namespace MLFramework.Amp
{
    /// <summary>
    /// Configuration settings for Automatic Mixed Precision
    /// </summary>
    public class AmpConfig
    {
        /// <summary>
        /// Gets or sets the target precision for AMP (FP16 or BF16)
        /// </summary>
        public DataType TargetPrecision { get; set; }

        /// <summary>
        /// Gets or sets whether to enable AMP globally
        /// </summary>
        public bool Enabled { get; set; }

        /// <summary>
        /// Gets or sets the default higher precision type (usually Float32)
        /// </summary>
        public DataType HigherPrecision { get; set; }

        /// <summary>
        /// Gets or sets whether to use view casting (zero-copy) when possible
        /// </summary>
        public bool UseViewCasting { get; set; }

        /// <summary>
        /// Gets or sets whether to enable kernel fusion across precision boundaries
        /// </summary>
        public bool EnableKernelFusion { get; set; }

        /// <summary>
        /// Creates a default AMP config with BF16 (recommended for Transformers)
        /// </summary>
        public static AmpConfig CreateDefault()
        {
            return new AmpConfig
            {
                TargetPrecision = DataType.BFloat16,
                Enabled = true,
                HigherPrecision = DataType.Float32,
                UseViewCasting = true,
                EnableKernelFusion = true
            };
        }

        /// <summary>
        /// Creates an AMP config with FP16 (better for older hardware)
        /// </summary>
        public static AmpConfig CreateFp16()
        {
            return new AmpConfig
            {
                TargetPrecision = DataType.Float16,
                Enabled = true,
                HigherPrecision = DataType.Float32,
                UseViewCasting = true,
                EnableKernelFusion = true
            };
        }

        /// <summary>
        /// Creates an AMP config with BF16 (better for dynamic range)
        /// </summary>
        public static AmpConfig CreateBf16()
        {
            return new AmpConfig
            {
                TargetPrecision = DataType.BFloat16,
                Enabled = true,
                HigherPrecision = DataType.Float32,
                UseViewCasting = true,
                EnableKernelFusion = true
            };
        }
    }
}
