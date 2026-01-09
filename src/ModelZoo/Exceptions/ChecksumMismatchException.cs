using System;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Exception thrown when the downloaded file's checksum does not match the expected checksum.
    /// </summary>
    public class ChecksumMismatchException : Exception
    {
        /// <summary>
        /// Path to the file that failed checksum validation.
        /// </summary>
        public string FilePath { get; }

        /// <summary>
        /// Expected checksum (SHA256 hash).
        /// </summary>
        public string ExpectedChecksum { get; }

        /// <summary>
        /// Actual checksum computed from the downloaded file.
        /// </summary>
        public string ActualChecksum { get; }

        /// <summary>
        /// Creates a new ChecksumMismatchException.
        /// </summary>
        /// <param name="filePath">Path to the file that failed checksum validation.</param>
        /// <param name="expectedChecksum">Expected checksum.</param>
        /// <param name="actualChecksum">Actual checksum.</param>
        public ChecksumMismatchException(string filePath, string expectedChecksum, string actualChecksum)
            : base($"Checksum mismatch for file '{filePath}'. Expected: {expectedChecksum}, Actual: {actualChecksum}")
        {
            FilePath = filePath;
            ExpectedChecksum = expectedChecksum;
            ActualChecksum = actualChecksum;
        }

        /// <summary>
        /// Creates a new ChecksumMismatchException with an inner exception.
        /// </summary>
        /// <param name="filePath">Path to the file that failed checksum validation.</param>
        /// <param name="expectedChecksum">Expected checksum.</param>
        /// <param name="actualChecksum">Actual checksum.</param>
        /// <param name="innerException">The exception that caused this exception.</param>
        public ChecksumMismatchException(string filePath, string expectedChecksum, string actualChecksum, Exception innerException)
            : base($"Checksum mismatch for file '{filePath}'. Expected: {expectedChecksum}, Actual: {actualChecksum}", innerException)
        {
            FilePath = filePath;
            ExpectedChecksum = expectedChecksum;
            ActualChecksum = actualChecksum;
        }
    }
}
