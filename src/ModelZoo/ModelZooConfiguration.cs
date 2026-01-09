using MLFramework.Core;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Configuration settings for the ModelZoo.
    /// </summary>
    public class ModelZooConfiguration
    {
        /// <summary>
        /// Gets or sets the default device for model operations.
        /// </summary>
        public Device DefaultDevice { get; set; }

        /// <summary>
        /// Gets or sets whether caching is enabled.
        /// </summary>
        public bool CacheEnabled { get; set; }

        /// <summary>
        /// Gets or sets whether auto-download is enabled.
        /// </summary>
        public bool AutoDownloadEnabled { get; set; }

        /// <summary>
        /// Gets or sets the default download timeout in milliseconds.
        /// </summary>
        public int DefaultDownloadTimeoutMs { get; set; }

        /// <summary>
        /// Creates a new ModelZooConfiguration with default settings.
        /// </summary>
        public ModelZooConfiguration()
        {
            DefaultDevice = Device.CreateCpu();
            CacheEnabled = true;
            AutoDownloadEnabled = true;
            DefaultDownloadTimeoutMs = 300000; // 5 minutes
        }

        /// <summary>
        /// Creates a new ModelZooConfiguration with specified settings.
        /// </summary>
        /// <param name="defaultDevice">The default device for model operations.</param>
        /// <param name="cacheEnabled">Whether caching is enabled.</param>
        /// <param name="autoDownloadEnabled">Whether auto-download is enabled.</param>
        /// <param name="defaultDownloadTimeoutMs">The default download timeout in milliseconds.</param>
        public ModelZooConfiguration(
            Device defaultDevice,
            bool cacheEnabled = true,
            bool autoDownloadEnabled = true,
            int defaultDownloadTimeoutMs = 300000)
        {
            DefaultDevice = defaultDevice ?? Device.CreateCpu();
            CacheEnabled = cacheEnabled;
            AutoDownloadEnabled = autoDownloadEnabled;
            DefaultDownloadTimeoutMs = defaultDownloadTimeoutMs;
        }
    }
}
