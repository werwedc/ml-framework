using System;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Configuration settings for the model cache manager.
    /// </summary>
    public class CacheConfiguration
    {
        /// <summary>
        /// Gets or sets the default cache root directory path.
        /// Default is OS-specific: ~/.ml-framework on Linux/Mac, %APPDATA%/MLFramework on Windows.
        /// </summary>
        public string CacheRootPath { get; set; }

        /// <summary>
        /// Gets or sets the maximum cache size in bytes.
        /// Default is 10GB.
        /// </summary>
        public long MaxCacheSizeBytes { get; set; }

        /// <summary>
        /// Gets or sets the maximum file age before pruning.
        /// Default is 30 days.
        /// </summary>
        public TimeSpan MaxFileAge { get; set; }

        /// <summary>
        /// Gets or sets whether to run cleanup on startup.
        /// Default is true.
        /// </summary>
        public bool CleanupOnStartup { get; set; }

        /// <summary>
        /// Gets or sets the timeout for file lock operations in milliseconds.
        /// Default is 5000ms.
        /// </summary>
        public int LockTimeoutMs { get; set; }

        /// <summary>
        /// Initializes a new instance of the CacheConfiguration class with default values.
        /// </summary>
        public CacheConfiguration()
        {
            CacheRootPath = GetDefaultCachePath();
            MaxCacheSizeBytes = 10L * 1024 * 1024 * 1024; // 10GB
            MaxFileAge = TimeSpan.FromDays(30);
            CleanupOnStartup = true;
            LockTimeoutMs = 5000;
        }

        /// <summary>
        /// Gets the default cache path based on the operating system.
        /// </summary>
        /// <returns>The default cache directory path.</returns>
        private static string GetDefaultCachePath()
        {
            if (Environment.OSVersion.Platform == PlatformID.Win32NT)
            {
                string appData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                return System.IO.Path.Combine(appData, "MLFramework", "ModelZoo");
            }
            else
            {
                string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                return System.IO.Path.Combine(home, ".ml-framework", "model-zoo");
            }
        }
    }
}
