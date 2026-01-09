using System;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Exception thrown when download fails after all retries and mirror attempts.
    /// </summary>
    public class DownloadFailedException : Exception
    {
        /// <summary>
        /// List of URLs that were attempted.
        /// </summary>
        public string[] AttemptedUrls { get; }

        /// <summary>
        /// Number of retries attempted.
        /// </summary>
        public int RetryCount { get; }

        /// <summary>
        /// Creates a new DownloadFailedException.
        /// </summary>
        /// <param name="attemptedUrls">List of URLs that were attempted.</param>
        /// <param name="retryCount">Number of retries attempted.</param>
        public DownloadFailedException(string[] attemptedUrls, int retryCount)
            : base($"Download failed after attempting {attemptedUrls.Length} URL(s) with {retryCount} retry/retries each.")
        {
            AttemptedUrls = attemptedUrls;
            RetryCount = retryCount;
        }

        /// <summary>
        /// Creates a new DownloadFailedException with an inner exception.
        /// </summary>
        /// <param name="attemptedUrls">List of URLs that were attempted.</param>
        /// <param name="retryCount">Number of retries attempted.</param>
        /// <param name="innerException">The exception that caused this exception.</param>
        public DownloadFailedException(string[] attemptedUrls, int retryCount, Exception innerException)
            : base($"Download failed after attempting {attemptedUrls.Length} URL(s) with {retryCount} retry/retries each.", innerException)
        {
            AttemptedUrls = attemptedUrls;
            RetryCount = retryCount;
        }

        /// <summary>
        /// Creates a new DownloadFailedException with a custom message.
        /// </summary>
        /// <param name="message">Custom error message.</param>
        /// <param name="attemptedUrls">List of URLs that were attempted.</param>
        /// <param name="retryCount">Number of retries attempted.</param>
        public DownloadFailedException(string message, string[] attemptedUrls, int retryCount)
            : base(message)
        {
            AttemptedUrls = attemptedUrls;
            RetryCount = retryCount;
        }
    }
}
