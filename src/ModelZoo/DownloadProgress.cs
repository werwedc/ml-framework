using System;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Tracks download state and provides progress information.
    /// </summary>
    public class DownloadProgress
    {
        /// <summary>
        /// Number of bytes downloaded so far.
        /// </summary>
        public long BytesDownloaded { get; set; }

        /// <summary>
        /// Total number of bytes to download.
        /// </summary>
        public long TotalBytes { get; set; }

        /// <summary>
        /// Current download speed in bytes per second.
        /// </summary>
        public double DownloadSpeed { get; set; }

        /// <summary>
        /// Estimated time remaining in seconds.
        /// </summary>
        public TimeSpan EstimatedTimeRemaining { get; set; }

        /// <summary>
        /// URL currently being downloaded (which mirror is being used).
        /// </summary>
        public string CurrentUrl { get; set; }

        /// <summary>
        /// Progress percentage (0.0 to 1.0).
        /// </summary>
        public double ProgressPercentage => TotalBytes > 0 ? (double)BytesDownloaded / TotalBytes : 0.0;

        /// <summary>
        /// Progress percentage as a human-readable string (0-100).
        /// </summary>
        public string ProgressPercentageText => $"{ProgressPercentage * 100:F1}%";

        /// <summary>
        /// Downloaded size formatted as human-readable string (e.g., "1.5 MB").
        /// </summary>
        public string BytesDownloadedText => FormatBytes(BytesDownloaded);

        /// <summary>
        /// Total size formatted as human-readable string (e.g., "500 MB").
        /// </summary>
        public string TotalBytesText => FormatBytes(TotalBytes);

        /// <summary>
        /// Download speed formatted as human-readable string (e.g., "5.2 MB/s").
        /// </summary>
        public string DownloadSpeedText => $"{FormatBytes((long)DownloadSpeed)}/s";

        /// <summary>
        /// Creates a new DownloadProgress instance.
        /// </summary>
        public DownloadProgress()
        {
        }

        /// <summary>
        /// Creates a new DownloadProgress instance with initial values.
        /// </summary>
        /// <param name="bytesDownloaded">Bytes downloaded so far.</param>
        /// <param name="totalBytes">Total bytes to download.</param>
        /// <param name="currentUrl">URL currently being downloaded.</param>
        public DownloadProgress(long bytesDownloaded, long totalBytes, string currentUrl = null)
        {
            BytesDownloaded = bytesDownloaded;
            TotalBytes = totalBytes;
            CurrentUrl = currentUrl;
        }

        /// <summary>
        /// Formats a byte count as a human-readable string.
        /// </summary>
        private static string FormatBytes(long bytes)
        {
            string[] sizes = { "B", "KB", "MB", "GB", "TB" };
            double len = bytes;
            int order = 0;

            while (len >= 1024 && order < sizes.Length - 1)
            {
                order++;
                len /= 1024;
            }

            return $"{len:0.##} {sizes[order]}";
        }
    }
}
