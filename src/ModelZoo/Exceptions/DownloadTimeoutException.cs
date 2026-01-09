using System;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Exception thrown when download operation times out.
    /// </summary>
    public class DownloadTimeoutException : Exception
    {
        /// <summary>
        /// URL that timed out.
        /// </summary>
        public string Url { get; }

        /// <summary>
        /// Timeout duration in milliseconds.
        /// </summary>
        public long TimeoutMs { get; }

        /// <summary>
        /// Number of bytes downloaded before timeout.
        /// </summary>
        public long BytesDownloaded { get; }

        /// <summary>
        /// Creates a new DownloadTimeoutException.
        /// </summary>
        /// <param name="url">URL that timed out.</param>
        /// <param name="timeoutMs">Timeout duration in milliseconds.</param>
        public DownloadTimeoutException(string url, long timeoutMs)
            : base($"Download from '{url}' timed out after {timeoutMs}ms.")
        {
            Url = url;
            TimeoutMs = timeoutMs;
            BytesDownloaded = 0;
        }

        /// <summary>
        /// Creates a new DownloadTimeoutException with bytes downloaded information.
        /// </summary>
        /// <param name="url">URL that timed out.</param>
        /// <param name="timeoutMs">Timeout duration in milliseconds.</param>
        /// <param name="bytesDownloaded">Number of bytes downloaded before timeout.</param>
        public DownloadTimeoutException(string url, long timeoutMs, long bytesDownloaded)
            : base($"Download from '{url}' timed out after {timeoutMs}ms. Downloaded {bytesDownloaded} bytes before timeout.")
        {
            Url = url;
            TimeoutMs = timeoutMs;
            BytesDownloaded = bytesDownloaded;
        }

        /// <summary>
        /// Creates a new DownloadTimeoutException with an inner exception.
        /// </summary>
        /// <param name="url">URL that timed out.</param>
        /// <param name="timeoutMs">Timeout duration in milliseconds.</param>
        /// <param name="innerException">The exception that caused this exception.</param>
        public DownloadTimeoutException(string url, long timeoutMs, Exception innerException)
            : base($"Download from '{url}' timed out after {timeoutMs}ms.", innerException)
        {
            Url = url;
            TimeoutMs = timeoutMs;
            BytesDownloaded = 0;
        }
    }
}
