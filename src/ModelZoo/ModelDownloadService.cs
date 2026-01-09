using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Service for downloading model files with checksum verification, resume capability, and progress reporting.
    /// </summary>
    public class ModelDownloadService : IDisposable
    {
        private readonly HttpClient _httpClient;
        private readonly int _defaultTimeoutMs;
        private readonly int _maxRetries;
        private readonly int _chunkSizeBytes;
        private bool _disposed;

        /// <summary>
        /// Creates a new ModelDownloadService with default settings.
        /// </summary>
        public ModelDownloadService()
            : this(TimeSpan.FromMinutes(5), 3, 1024 * 1024) // 5 min timeout, 3 retries, 1MB chunks
        {
        }

        /// <summary>
        /// Creates a new ModelDownloadService with custom settings.
        /// </summary>
        /// <param name="timeout">Default timeout for download operations.</param>
        /// <param name="maxRetries">Maximum number of retry attempts for failed downloads.</param>
        /// <param name="chunkSizeBytes">Size of download chunks for progress reporting.</param>
        public ModelDownloadService(TimeSpan timeout, int maxRetries, int chunkSizeBytes)
        {
            _defaultTimeoutMs = (int)timeout.TotalMilliseconds;
            _maxRetries = maxRetries;
            _chunkSizeBytes = chunkSizeBytes;

            _httpClient = new HttpClient(new HttpClientHandler
            {
                AllowAutoRedirect = true,
                MaxAutomaticRedirections = 10
            });

            _httpClient.Timeout = timeout;
        }

        /// <summary>
        /// Downloads a model file with checksum verification.
        /// </summary>
        /// <param name="url">URL to download from.</param>
        /// <param name="destinationPath">Path where the file should be saved.</param>
        /// <param name="expectedSha256">Expected SHA256 checksum for verification.</param>
        /// <param name="progress">Optional progress reporter (0.0 to 1.0).</param>
        /// <param name="cancellationToken">Optional cancellation token.</param>
        public async Task DownloadModelAsync(
            string url,
            string destinationPath,
            string expectedSha256,
            IProgress<double> progress = null,
            CancellationToken cancellationToken = default)
        {
            await DownloadWithRetryAsync(
                url,
                destinationPath,
                expectedSha256,
                progress,
                cancellationToken,
                enableResume: false);
        }

        /// <summary>
        /// Downloads a model file with checksum verification and resume capability.
        /// </summary>
        /// <param name="url">URL to download from.</param>
        /// <param name="destinationPath">Path where the file should be saved.</param>
        /// <param name="expectedSha256">Expected SHA256 checksum for verification.</param>
        /// <param name="progress">Optional progress reporter (0.0 to 1.0).</param>
        /// <param name="cancellationToken">Optional cancellation token.</param>
        public async Task DownloadWithResumeAsync(
            string url,
            string destinationPath,
            string expectedSha256,
            IProgress<double> progress = null,
            CancellationToken cancellationToken = default)
        {
            await DownloadWithRetryAsync(
                url,
                destinationPath,
                expectedSha256,
                progress,
                cancellationToken,
                enableResume: true);
        }

        /// <summary>
        /// Downloads from multiple mirrors with fallback.
        /// </summary>
        /// <param name="urls">List of mirror URLs to try in order.</param>
        /// <param name="destinationPath">Path where the file should be saved.</param>
        /// <param name="expectedSha256">Expected SHA256 checksum for verification.</param>
        /// <param name="progress">Optional progress reporter (0.0 to 1.0).</param>
        /// <param name="cancellationToken">Optional cancellation token.</param>
        public async Task DownloadFromMirrorsAsync(
            string[] urls,
            string destinationPath,
            string expectedSha256,
            IProgress<double> progress = null,
            CancellationToken cancellationToken = default)
        {
            if (urls == null || urls.Length == 0)
            {
                throw new ArgumentException("URLs array cannot be null or empty.", nameof(urls));
            }

            List<Exception> exceptions = new List<Exception>();

            foreach (string url in urls)
            {
                try
                {
                    await DownloadWithRetryAsync(
                        url,
                        destinationPath,
                        expectedSha256,
                        progress,
                        cancellationToken,
                        enableResume: true);

                    // Success - return without trying other mirrors
                    return;
                }
                catch (Exception ex) when (!(ex is OperationCanceledException))
                {
                    exceptions.Add(ex);
                    // Try next mirror
                }
            }

            // All mirrors failed
            throw new DownloadFailedException(urls, _maxRetries,
                new AggregateException("All mirror URLs failed.", exceptions));
        }

        /// <summary>
        /// Verifies the SHA256 checksum of a downloaded file.
        /// </summary>
        /// <param name="filePath">Path to the file to verify.</param>
        /// <param name="expectedSha256">Expected SHA256 checksum.</param>
        /// <returns>True if checksum matches, false otherwise.</returns>
        public bool VerifyChecksum(string filePath, string expectedSha256)
        {
            if (!File.Exists(filePath))
            {
                return false;
            }

            string actualChecksum = ComputeSha256Hash(filePath);
            return string.Equals(actualChecksum, expectedSha256, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Computes the SHA256 hash of a file.
        /// </summary>
        /// <param name="filePath">Path to the file.</param>
        /// <returns>SHA256 hash as a hexadecimal string.</returns>
        public string ComputeSha256Hash(string filePath)
        {
            using (var sha256 = SHA256.Create())
            using (var stream = File.OpenRead(filePath))
            {
                byte[] hash = sha256.ComputeHash(stream);
                return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
            }
        }

        /// <summary>
        /// Internal download method with retry logic.
        /// </summary>
        private async Task DownloadWithRetryAsync(
            string url,
            string destinationPath,
            string expectedSha256,
            IProgress<double> progress,
            CancellationToken cancellationToken,
            bool enableResume)
        {
            if (string.IsNullOrWhiteSpace(url))
            {
                throw new ArgumentException("URL cannot be null or empty.", nameof(url));
            }

            if (string.IsNullOrWhiteSpace(destinationPath))
            {
                throw new ArgumentException("Destination path cannot be null or empty.", nameof(destinationPath));
            }

            // Ensure destination directory exists
            string directory = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            Exception lastException = null;

            for (int attempt = 0; attempt < _maxRetries; attempt++)
            {
                try
                {
                    if (enableResume && File.Exists(destinationPath))
                    {
                        await DownloadWithRangeAsync(url, destinationPath, progress, cancellationToken);
                    }
                    else
                    {
                        await DownloadFullAsync(url, destinationPath, progress, cancellationToken);
                    }

                    // Verify checksum after download
                    if (!string.IsNullOrWhiteSpace(expectedSha256))
                    {
                        string actualChecksum = ComputeSha256Hash(destinationPath);
                        if (!string.Equals(actualChecksum, expectedSha256, StringComparison.OrdinalIgnoreCase))
                        {
                            // Delete corrupted file
                            File.Delete(destinationPath);
                            throw new ChecksumMismatchException(destinationPath, expectedSha256, actualChecksum);
                        }
                    }

                    // Success!
                    return;
                }
                catch (OperationCanceledException)
                {
                    // Re-throw cancellation
                    throw;
                }
                catch (HttpRequestException ex) when (ex.StatusCode == null || !IsSuccessStatusCode(ex.StatusCode.Value))
                {
                    lastException = ex;
                }
                catch (TimeoutException ex)
                {
                    lastException = ex;
                }
                catch (Exception ex) when (!(ex is ChecksumMismatchException))
                {
                    lastException = ex;
                }

                // Exponential backoff before retry
                if (attempt < _maxRetries - 1)
                {
                    int delayMs = (int)Math.Pow(2, attempt) * 1000; // 1s, 2s, 4s, ...
                    await Task.Delay(delayMs, cancellationToken);
                }
            }

            // All retries failed
            throw new DownloadFailedException(new[] { url }, _maxRetries, lastException);
        }

        /// <summary>
        /// Downloads the full file from URL.
        /// </summary>
        private async Task DownloadFullAsync(
            string url,
            string destinationPath,
            IProgress<double> progress,
            CancellationToken cancellationToken)
        {
            using (var response = await _httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
            {
                response.EnsureSuccessStatusCode();

                long totalBytes = response.Content.Headers.ContentLength ?? 0;
                long downloadedBytes = 0;

                using (var contentStream = await response.Content.ReadAsStreamAsync())
                using (var fileStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    byte[] buffer = new byte[_chunkSizeBytes];
                    int bytesRead;

                    while ((bytesRead = await contentStream.ReadAsync(buffer, 0, buffer.Length, cancellationToken)) > 0)
                    {
                        await fileStream.WriteAsync(buffer, 0, bytesRead, cancellationToken);
                        downloadedBytes += bytesRead;

                        if (progress != null && totalBytes > 0)
                        {
                            progress.Report((double)downloadedBytes / totalBytes);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Downloads with range support for resume capability.
        /// </summary>
        private async Task DownloadWithRangeAsync(
            string url,
            string destinationPath,
            IProgress<double> progress,
            CancellationToken cancellationToken)
        {
            // Get file size to determine if we need to resume
            long existingFileSize = new FileInfo(destinationPath).Length;

            using (var request = new HttpRequestMessage(HttpMethod.Get, url))
            {
                // Set Range header to resume from existing position
                request.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(existingFileSize, null);

                using (var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
                {
                    // Server may return 206 (Partial Content) or 200 (OK) if range not supported
                    if (!response.IsSuccessStatusCode &&
                        response.StatusCode != System.Net.HttpStatusCode.PartialContent)
                    {
                        // Resume not supported, delete and start fresh
                        File.Delete(destinationPath);
                        await DownloadFullAsync(url, destinationPath, progress, cancellationToken);
                        return;
                    }

                    long totalBytes = GetContentLength(response.Headers);
                    long startPosition = existingFileSize;

                    if (response.StatusCode == System.Net.HttpStatusCode.OK)
                    {
                        // Server doesn't support range, restart from beginning
                        startPosition = 0;
                        File.Delete(destinationPath);
                    }

                    using (var contentStream = await response.Content.ReadAsStreamAsync())
                    using (var fileStream = new FileStream(destinationPath,
                        startPosition == 0 ? FileMode.Create : FileMode.Append,
                        FileAccess.Write,
                        FileShare.None))
                    {
                        if (startPosition > 0)
                        {
                            fileStream.Seek(startPosition, SeekOrigin.Begin);
                        }

                        byte[] buffer = new byte[_chunkSizeBytes];
                        int bytesRead;
                        long downloadedBytes = startPosition;

                        while ((bytesRead = await contentStream.ReadAsync(buffer, 0, buffer.Length, cancellationToken)) > 0)
                        {
                            await fileStream.WriteAsync(buffer, 0, bytesRead, cancellationToken);
                            downloadedBytes += bytesRead;

                            if (progress != null && totalBytes > 0)
                            {
                                progress.Report((double)downloadedBytes / totalBytes);
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Gets the total content length from response headers.
        /// </summary>
        private long GetContentLength(System.Net.Http.Headers.HttpResponseHeaders headers)
        {
            if (headers.ContentLength.HasValue)
            {
                return headers.ContentLength.Value;
            }

            // Try Content-Range header for partial content
            if (headers.Contains("Content-Range"))
            {
                var contentRange = headers.GetValues("Content-Range").FirstOrDefault();
                if (!string.IsNullOrEmpty(contentRange))
                {
                    // Format: "bytes start-end/total"
                    var parts = contentRange.Split('/');
                    if (parts.Length == 2 && long.TryParse(parts[1], out long total))
                    {
                        return total;
                    }
                }
            }

            return 0;
        }

        /// <summary>
        /// Checks if an HTTP status code indicates success.
        /// </summary>
        private bool IsSuccessStatusCode(System.Net.HttpStatusCode statusCode)
        {
            return ((int)statusCode >= 200 && (int)statusCode < 300);
        }

        /// <summary>
        /// Releases unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Releases unmanaged resources and optionally releases managed resources.
        /// </summary>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _httpClient?.Dispose();
                }

                _disposed = true;
            }
        }
    }
}
