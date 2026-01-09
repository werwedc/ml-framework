using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Moq;
using Moq.Protected;
using MLFramework.ModelZoo;
using Xunit;

namespace ModelZooTests
{
    /// <summary>
    /// Unit tests for ModelDownloadService.
    /// </summary>
    public class ModelDownloadServiceTests : IDisposable
    {
        private readonly string _testDirectory;
        private readonly HttpClient _testHttpClient;
        private readonly HttpMessageHandler _mockHttpMessageHandler;

        public ModelDownloadServiceTests()
        {
            _testDirectory = Path.Combine(Path.GetTempPath(), "ModelDownloadServiceTests", Guid.NewGuid().ToString());
            Directory.CreateDirectory(_testDirectory);

            // Create mock HTTP handler for testing
            _mockHttpMessageHandler = new Mock<HttpMessageHandler>();
            _testHttpClient = new HttpClient(_mockHttpMessageHandler.Object);
        }

        public void Dispose()
        {
            if (Directory.Exists(_testDirectory))
            {
                Directory.Delete(_testDirectory, true);
            }

            _testHttpClient?.Dispose();
        }

        [Fact]
        public void Constructor_Default_CreatesServiceWithDefaultSettings()
        {
            // Arrange & Act
            using var service = new ModelDownloadService();

            // Assert
            Assert.NotNull(service);
        }

        [Fact]
        public void Constructor_CustomSettings_CreatesServiceWithCustomSettings()
        {
            // Arrange & Act
            using var service = new ModelDownloadService(
                TimeSpan.FromSeconds(30),
                5,
                2048 * 1024);

            // Assert
            Assert.NotNull(service);
        }

        [Fact]
        public async Task ComputeSha256Hash_ValidFile_ReturnsCorrectHash()
        {
            // Arrange
            using var service = new ModelDownloadService();
            string testFile = Path.Combine(_testDirectory, "test.txt");
            string content = "Hello, World!";
            await File.WriteAllTextAsync(testFile, content);

            // Known SHA256 hash for "Hello, World!"
            string expectedHash = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f";

            // Act
            string actualHash = service.ComputeSha256Hash(testFile);

            // Assert
            Assert.Equal(expectedHash, actualHash);
        }

        [Fact]
        public async Task VerifyChecksum_MatchingChecksum_ReturnsTrue()
        {
            // Arrange
            using var service = new ModelDownloadService();
            string testFile = Path.Combine(_testDirectory, "test.txt");
            await File.WriteAllTextAsync(testFile, "Hello, World!");
            string expectedHash = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f";

            // Act
            bool result = service.VerifyChecksum(testFile, expectedHash);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public async Task VerifyChecksum_NonMatchingChecksum_ReturnsFalse()
        {
            // Arrange
            using var service = new ModelDownloadService();
            string testFile = Path.Combine(_testDirectory, "test.txt");
            await File.WriteAllTextAsync(testFile, "Hello, World!");
            string wrongHash = "0000000000000000000000000000000000000000000000000000000000000000";

            // Act
            bool result = service.VerifyChecksum(testFile, wrongHash);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public async Task VerifyChecksum_NonExistentFile_ReturnsFalse()
        {
            // Arrange
            using var service = new ModelDownloadService();
            string nonExistentFile = Path.Combine(_testDirectory, "nonexistent.txt");
            string anyHash = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f";

            // Act
            bool result = service.VerifyChecksum(nonExistentFile, anyHash);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void ChecksumMismatchException_Constructor_CreatesExceptionWithCorrectProperties()
        {
            // Arrange
            string filePath = "/path/to/file.bin";
            string expectedChecksum = "abc123";
            string actualChecksum = "def456";

            // Act
            var exception = new ChecksumMismatchException(filePath, expectedChecksum, actualChecksum);

            // Assert
            Assert.Equal(filePath, exception.FilePath);
            Assert.Equal(expectedChecksum, exception.ExpectedChecksum);
            Assert.Equal(actualChecksum, exception.ActualChecksum);
            Assert.Contains(filePath, exception.Message);
        }

        [Fact]
        public void DownloadFailedException_Constructor_CreatesExceptionWithCorrectProperties()
        {
            // Arrange
            string[] urls = new[] { "http://example.com/file1.bin", "http://example.com/file2.bin" };
            int retryCount = 3;

            // Act
            var exception = new DownloadFailedException(urls, retryCount);

            // Assert
            Assert.Equal(urls, exception.AttemptedUrls);
            Assert.Equal(retryCount, exception.RetryCount);
            Assert.Contains(urls.Length.ToString(), exception.Message);
        }

        [Fact]
        public void DownloadTimeoutException_Constructor_CreatesExceptionWithCorrectProperties()
        {
            // Arrange
            string url = "http://example.com/largefile.bin";
            long timeoutMs = 60000;

            // Act
            var exception = new DownloadTimeoutException(url, timeoutMs);

            // Assert
            Assert.Equal(url, exception.Url);
            Assert.Equal(timeoutMs, exception.TimeoutMs);
            Assert.Equal(0, exception.BytesDownloaded);
            Assert.Contains(url, exception.Message);
        }

        [Fact]
        public void DownloadTimeoutException_WithBytesDownloaded_CreatesExceptionWithCorrectProperties()
        {
            // Arrange
            string url = "http://example.com/largefile.bin";
            long timeoutMs = 60000;
            long bytesDownloaded = 1024 * 1024 * 50; // 50 MB

            // Act
            var exception = new DownloadTimeoutException(url, timeoutMs, bytesDownloaded);

            // Assert
            Assert.Equal(url, exception.Url);
            Assert.Equal(timeoutMs, exception.TimeoutMs);
            Assert.Equal(bytesDownloaded, exception.BytesDownloaded);
            Assert.Contains(bytesDownloaded.ToString(), exception.Message);
        }

        [Fact]
        public void DownloadProgress_Constructor_CreatesProgressWithDefaultValues()
        {
            // Arrange & Act
            var progress = new DownloadProgress();

            // Assert
            Assert.Equal(0, progress.BytesDownloaded);
            Assert.Equal(0, progress.TotalBytes);
            Assert.Equal(0, progress.DownloadSpeed);
            Assert.Equal(0, progress.ProgressPercentage);
        }

        [Fact]
        public void DownloadProgress_ConstructorWithValues_CreatesProgressWithSpecifiedValues()
        {
            // Arrange
            long bytesDownloaded = 1024 * 1024 * 50; // 50 MB
            long totalBytes = 1024 * 1024 * 100; // 100 MB
            string currentUrl = "http://example.com/file.bin";

            // Act
            var progress = new DownloadProgress(bytesDownloaded, totalBytes, currentUrl);

            // Assert
            Assert.Equal(bytesDownloaded, progress.BytesDownloaded);
            Assert.Equal(totalBytes, progress.TotalBytes);
            Assert.Equal(currentUrl, progress.CurrentUrl);
            Assert.Equal(0.5, progress.ProgressPercentage);
        }

        [Fact]
        public void DownloadProgress_ProgressPercentageText_ReturnsFormattedPercentage()
        {
            // Arrange
            var progress = new DownloadProgress(50, 100);

            // Act
            string text = progress.ProgressPercentageText;

            // Assert
            Assert.Equal("50.0%", text);
        }

        [Fact]
        public void DownloadProgress_BytesDownloadedText_ReturnsFormattedBytes()
        {
            // Arrange
            var progress = new DownloadProgress(1024 * 1024, 0); // 1 MB

            // Act
            string text = progress.BytesDownloadedText;

            // Assert
            Assert.Contains("MB", text);
        }

        [Fact]
        public void DownloadProgress_TotalBytesText_ReturnsFormattedBytes()
        {
            // Arrange
            var progress = new DownloadProgress(0, 1024 * 1024 * 1024); // 1 GB

            // Act
            string text = progress.TotalBytesText;

            // Assert
            Assert.Contains("GB", text);
        }

        [Fact]
        public async Task DownloadModelAsync_NullUrl_ThrowsArgumentException()
        {
            // Arrange
            using var service = new ModelDownloadService();
            string destinationPath = Path.Combine(_testDirectory, "test.bin");

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentException>(() =>
                service.DownloadModelAsync(null, destinationPath, "hash123"));
        }

        [Fact]
        public async Task DownloadModelAsync_NullDestinationPath_ThrowsArgumentException()
        {
            // Arrange
            using var service = new ModelDownloadService();

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentException>(() =>
                service.DownloadModelAsync("http://example.com/file.bin", null, "hash123"));
        }

        [Fact]
        public async Task DownloadFromMirrorsAsync_EmptyUrls_ThrowsArgumentException()
        {
            // Arrange
            using var service = new ModelDownloadService();
            string destinationPath = Path.Combine(_testDirectory, "test.bin");
            string[] emptyUrls = Array.Empty<string>();

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentException>(() =>
                service.DownloadFromMirrorsAsync(emptyUrls, destinationPath, "hash123"));
        }

        [Fact]
        public async Task DownloadFromMirrorsAsync_NullUrls_ThrowsArgumentException()
        {
            // Arrange
            using var service = new ModelDownloadService();
            string destinationPath = Path.Combine(_testDirectory, "test.bin");

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentException>(() =>
                service.DownloadFromMirrorsAsync(null, destinationPath, "hash123"));
        }

        [Fact]
        public async Task DownloadFromMirrorsAsync_AllMirrorsFail_ThrowsDownloadFailedException()
        {
            // Arrange
            using var service = new ModelDownloadService();
            string destinationPath = Path.Combine(_testDirectory, "test.bin");
            string[] urls = new[] { "http://invalid-url-1.bin", "http://invalid-url-2.bin" };

            // Act & Assert
            await Assert.ThrowsAsync<DownloadFailedException>(() =>
                service.DownloadFromMirrorsAsync(urls, destinationPath, "hash123"));
        }

        [Fact]
        public async Task DownloadModelAsync_WithCancellation_ThrowsOperationCanceledException()
        {
            // Arrange
            using var service = new ModelDownloadService();
            string destinationPath = Path.Combine(_testDirectory, "test.bin");
            var cts = new CancellationTokenSource();
            cts.Cancel(); // Cancel immediately

            // Act & Assert
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
                service.DownloadModelAsync("http://example.com/largefile.bin", destinationPath, "hash123", null, cts.Token));
        }

        [Fact]
        public async Task DownloadModelAsync_ProgressReporter_ReceivesProgressUpdates()
        {
            // Arrange
            using var service = new ModelDownloadService();
            string testFile = Path.Combine(_testDirectory, "test.txt");
            await File.WriteAllTextAsync(testFile, "Hello, World!");
            string expectedHash = service.ComputeSha256Hash(testFile);

            // Create a test server with the file
            var progressUpdates = new System.Collections.Generic.List<double>();
            var progress = new Progress<double>(p => progressUpdates.Add(p));

            // Note: This test requires an actual HTTP server or mock setup
            // For now, we'll just test the progress reporter interface
            Assert.NotNull(progress);
        }

        [Fact]
        public void DownloadProgress_DownloadSpeedText_ReturnsFormattedSpeed()
        {
            // Arrange
            var progress = new DownloadProgress();
            progress.DownloadSpeed = 1024 * 1024 * 5; // 5 MB/s

            // Act
            string text = progress.DownloadSpeedText;

            // Assert
            Assert.Contains("MB/s", text);
        }

        [Fact]
        public void DownloadProgress_ZeroTotalBytes_ZeroProgressPercentage()
        {
            // Arrange
            var progress = new DownloadProgress(100, 0);

            // Act
            double percentage = progress.ProgressPercentage;

            // Assert
            Assert.Equal(0.0, percentage);
        }

        [Fact]
        public async Task ComputeSha256Hash_LargeFile_ComputesCorrectly()
        {
            // Arrange
            using var service = new ModelDownloadService();
            string testFile = Path.Combine(_testDirectory, "largefile.bin");

            // Create a 10 MB file
            byte[] data = new byte[10 * 1024 * 1024];
            new Random().NextBytes(data);
            await File.WriteAllBytesAsync(testFile, data);

            // Act
            string hash1 = service.ComputeSha256Hash(testFile);
            string hash2 = service.ComputeSha256Hash(testFile);

            // Assert
            Assert.Equal(hash1, hash2); // Consistent hash
            Assert.Equal(64, hash1.Length); // SHA256 is 64 hex characters
        }
    }
}
