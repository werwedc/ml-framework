# Spec: Model Download Service

## Overview
Implement a service to download model files from URLs with checksum verification and resume capability.

## Requirements

### 1. ModelDownloadService Class
Download operations:
- `DownloadModel(string url, string destinationPath, string expectedSha256, IProgress<double> progress = null)`: Download with checksum verification
- `DownloadWithResume(string url, string destinationPath, string expectedSha256, IProgress<double> progress = null)`: Support resuming interrupted downloads
- `VerifyChecksum(string filePath, string expectedSha256)`: Validate downloaded file

### 2. Download Features
- HTTP/HTTPS support with automatic redirect handling
- Timeout configuration (default: 5 minutes)
- Chunked downloading (e.g., 1MB chunks for progress reporting)
- Checksum validation after download
- Automatic retry on failure (max 3 retries with exponential backoff)
- Resume support using HTTP Range headers
- Multi-mirror fallback (try mirrors in order if primary fails)

### 3. Progress Reporting
- Report download progress (0.0 to 1.0)
- Report download speed (bytes/second)
- Report estimated time remaining

### 4. DownloadProgress Class
Track download state:
- BytesDownloaded
- TotalBytes
- DownloadSpeed
- EstimatedTimeRemaining
- CurrentUrl (which mirror is being used)

### 5. DownloadException Types
Define specific exceptions:
- `ChecksumMismatchException`: SHA256 verification failed
- `DownloadFailedException`: All mirrors failed
- `DownloadTimeoutException`: Download timed out
- `DownloadInterruptedException`: Download was cancelled

### 6. Unit Tests
Test cases for:
- Successful download with valid checksum
- Checksum mismatch detection
- Resume interrupted download
- Mirror fallback behavior
- Progress reporting accuracy
- Timeout handling
- Cancel download (with cancellation token)
- Edge cases (zero-byte files, huge files)

## Files to Create
- `src/ModelZoo/ModelDownloadService.cs`
- `src/ModelZoo/DownloadProgress.cs`
- `src/ModelZoo/Exceptions/ChecksumMismatchException.cs`
- `src/ModelZoo/Exceptions/DownloadFailedException.cs`
- `src/ModelZoo/Exceptions/DownloadTimeoutException.cs`
- `tests/ModelZooTests/ModelDownloadServiceTests.cs`

## Dependencies
- System.Net.Http (for HTTP operations)
- System.IO.Hashing (for SHA256)
- System.Threading.Tasks (for async operations)

## Success Criteria
- Can download models > 10GB reliably
- Resume functionality works after interruption
- Checksum validation prevents corrupted downloads
- Mirror fallback provides 99.9% availability
- Test coverage > 85%
