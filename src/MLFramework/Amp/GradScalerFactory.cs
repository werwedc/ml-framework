using MLFramework.Amp;
using MLFramework.Optimizers.MixedPrecision;

namespace MLFramework.Amp
{
    /// <summary>
    /// Factory for creating GradScaler instances with common configurations
    /// </summary>
    public static class GradScalerFactory
    {
        /// <summary>
        /// Creates a default dynamic loss scaler (recommended for most use cases)
        /// </summary>
        public static GradScaler CreateDefault()
        {
            return new GradScaler(
                initialScale: 65536.0f,
                growthFactor: 2.0f,
                backoffFactor: 0.5f,
                growthInterval: 2000,
                minScale: 1.0f,
                maxScale: 16777216.0f,
                enabled: true);
        }

        /// <summary>
        /// Creates a static loss scaler with moderate scale
        /// </summary>
        public static GradScaler CreateStatic()
        {
            return new GradScaler(scale: 65536.0f, enabled: true);
        }

        /// <summary>
        /// Creates a static loss scaler with custom scale
        /// </summary>
        /// <param name="scale">The constant scaling factor</param>
        public static GradScaler CreateStatic(float scale)
        {
            return new GradScaler(scale: scale, enabled: true);
        }

        /// <summary>
        /// Creates a dynamic loss scaler with conservative settings
        /// </summary>
        public static GradScaler CreateConservative()
        {
            return new GradScaler(
                initialScale: 32768.0f,
                growthFactor: 2.0f,
                backoffFactor: 0.5f,
                growthInterval: 5000,
                minScale: 1.0f,
                maxScale: 16777216.0f,
                enabled: true);
        }

        /// <summary>
        /// Creates a dynamic loss scaler with aggressive settings
        /// </summary>
        public static GradScaler CreateAggressive()
        {
            return new GradScaler(
                initialScale: 65536.0f,
                growthFactor: 2.0f,
                backoffFactor: 0.5f,
                growthInterval: 1000,
                minScale: 1.0f,
                maxScale: 16777216.0f,
                enabled: true);
        }

        /// <summary>
        /// Creates a loss scaler optimized for FP16 training
        /// </summary>
        public static GradScaler CreateForFP16()
        {
            return new GradScaler(
                initialScale: 65536.0f,
                growthFactor: 2.0f,
                backoffFactor: 0.5f,
                growthInterval: 2000,
                minScale: 1.0f,
                maxScale: 16777216.0f,
                enabled: true);
        }

        /// <summary>
        /// Creates a loss scaler optimized for BF16 training
        /// </summary>
        public static GradScaler CreateForBF16()
        {
            // BF16 has a wider range than FP16, so we can use a smaller scale
            // or even static scaling
            return new GradScaler(
                initialScale: 8.0f,
                growthFactor: 2.0f,
                backoffFactor: 0.5f,
                growthInterval: 2000,
                minScale: 1.0f,
                maxScale: 65536.0f,
                enabled: true);
        }

        /// <summary>
        /// Creates a loss scaler with custom configuration
        /// </summary>
        /// <param name="config">The dynamic scaler configuration</param>
        public static GradScaler CreateFromConfig(DynamicScalerConfig config)
        {
            if (config == null)
                throw new System.ArgumentNullException(nameof(config));

            return new GradScaler(
                initialScale: config.InitialScale,
                growthFactor: config.GrowthFactor,
                backoffFactor: config.BackoffFactor,
                growthInterval: config.GrowthInterval,
                minScale: config.MinScale,
                maxScale: config.MaxScale,
                enabled: true);
        }
    }
}
