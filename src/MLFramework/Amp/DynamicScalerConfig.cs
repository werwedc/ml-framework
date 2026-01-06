namespace MLFramework.Amp
{
    /// <summary>
    /// Configuration for dynamic loss scaler
    /// </summary>
    public class DynamicScalerConfig
    {
        /// <summary>
        /// Gets or sets the initial scale factor
        /// </summary>
        public float InitialScale { get; set; }

        /// <summary>
        /// Gets or sets the growth factor for increasing scale
        /// </summary>
        public float GrowthFactor { get; set; }

        /// <summary>
        /// Gets or sets the backoff factor for decreasing scale
        /// </summary>
        public float BackoffFactor { get; set; }

        /// <summary>
        /// Gets or sets the growth interval
        /// </summary>
        public int GrowthInterval { get; set; }

        /// <summary>
        /// Gets or sets the minimum scale
        /// </summary>
        public float MinScale { get; set; }

        /// <summary>
        /// Gets or sets the maximum scale
        /// </summary>
        public float MaxScale { get; set; }

        /// <summary>
        /// Creates a default configuration
        /// </summary>
        public static DynamicScalerConfig CreateDefault()
        {
            return new DynamicScalerConfig
            {
                InitialScale = 65536.0f,
                GrowthFactor = 2.0f,
                BackoffFactor = 0.5f,
                GrowthInterval = 2000,
                MinScale = 1.0f,
                MaxScale = 16777216.0f
            };
        }

        /// <summary>
        /// Creates a conservative configuration (slower scale increase)
        /// </summary>
        public static DynamicScalerConfig CreateConservative()
        {
            return new DynamicScalerConfig
            {
                InitialScale = 32768.0f,
                GrowthFactor = 2.0f,
                BackoffFactor = 0.5f,
                GrowthInterval = 5000,
                MinScale = 1.0f,
                MaxScale = 16777216.0f
            };
        }

        /// <summary>
        /// Creates an aggressive configuration (faster scale increase)
        /// </summary>
        public static DynamicScalerConfig CreateAggressive()
        {
            return new DynamicScalerConfig
            {
                InitialScale = 65536.0f,
                GrowthFactor = 2.0f,
                BackoffFactor = 0.5f,
                GrowthInterval = 1000,
                MinScale = 1.0f,
                MaxScale = 16777216.0f
            };
        }
    }
}
