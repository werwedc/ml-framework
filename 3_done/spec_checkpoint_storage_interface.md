# Spec: Checkpoint Storage Backend Interface

## Overview
Define a pluggable storage interface for distributed checkpointing that supports local filesystem, cloud storage (S3, GCS, Azure Blob), and async I/O operations.

## Scope
- 30-45 minutes coding time
- Focus on interface design and base implementation
- Target: `src/MLFramework/Checkpointing/Storage/`

## Classes

### 1. ICheckpointStorage (Interface)
```csharp
public interface ICheckpointStorage
{
    /// <summary>
    /// Write data to storage asynchronously
    /// </summary>
    Task WriteAsync(string path, byte[] data, CancellationToken cancellationToken = default);

    /// <summary>
    /// Read data from storage asynchronously
    /// </summary>
    Task<byte[]> ReadAsync(string path, CancellationToken cancellationToken = default);

    /// <summary>
    /// Check if a file exists in storage
    /// </summary>
    Task<bool> ExistsAsync(string path, CancellationToken cancellationToken = default);

    /// <summary>
    /// Delete a file from storage
    /// </summary>
    Task DeleteAsync(string path, CancellationToken cancellationToken = default);

    /// <summary>
    /// Get storage metadata (size, last modified, etc.)
    /// </summary>
    Task<StorageMetadata> GetMetadataAsync(string path, CancellationToken cancellationToken = default);
}
```

### 2. StorageMetadata (Data Class)
```csharp
public class StorageMetadata
{
    public long Size { get; set; }
    public DateTime LastModified { get; set; }
    public Dictionary<string, string> AdditionalInfo { get; set; }
}
```

### 3. LocalFileSystemStorage (Concrete Implementation)
```csharp
public class LocalFileSystemStorage : ICheckpointStorage
{
    private readonly string _basePath;

    public LocalFileSystemStorage(string basePath)
    {
        _basePath = basePath;
        Directory.CreateDirectory(_basePath);
    }

    public async Task WriteAsync(string path, byte[] data, CancellationToken cancellationToken)
    {
        var fullPath = Path.Combine(_basePath, path);
        var directory = Path.GetDirectoryName(fullPath);
        Directory.CreateDirectory(directory);
        await File.WriteAllBytesAsync(fullPath, data, cancellationToken);
    }

    public async Task<byte[]> ReadAsync(string path, CancellationToken cancellationToken)
    {
        var fullPath = Path.Combine(_basePath, path);
        return await File.ReadAllBytesAsync(fullPath, cancellationToken);
    }

    public async Task<bool> ExistsAsync(string path, CancellationToken cancellationToken)
    {
        var fullPath = Path.Combine(_basePath, path);
        return await Task.Run(() => File.Exists(fullPath), cancellationToken);
    }

    public Task DeleteAsync(string path, CancellationToken cancellationToken)
    {
        var fullPath = Path.Combine(_basePath, path);
        File.Delete(fullPath);
        return Task.CompletedTask;
    }

    public async Task<StorageMetadata> GetMetadataAsync(string path, CancellationToken cancellationToken)
    {
        var fullPath = Path.Combine(_basePath, path);
        var info = new FileInfo(fullPath);
        return new StorageMetadata
        {
            Size = info.Length,
            LastModified = info.LastWriteTimeUtc
        };
    }
}
```

### 4. StorageOptions (Configuration)
```csharp
public class StorageOptions
{
    public string Provider { get; set; } // "local", "s3", "gcs", "azure"
    public Dictionary<string, string> ConnectionSettings { get; set; }
}
```

### 5. StorageFactory (Factory Pattern)
```csharp
public static class StorageFactory
{
    public static ICheckpointStorage Create(StorageOptions options)
    {
        return options.Provider.ToLower() switch
        {
            "local" => new LocalFileSystemStorage(
                options.ConnectionSettings["basePath"]),
            // Future: s3, gcs, azure implementations
            _ => throw new ArgumentException($"Unknown storage provider: {options.Provider}")
        };
    }
}
```

## Integration Points
- Used by: `DistributedCheckpointCoordinator`
- Depends on: System.IO, System.Threading.Tasks

## Testing Requirements
- Test write/read roundtrip
- Test file existence checks
- Test metadata retrieval
- Test path handling with nested directories
- Test cancellation token propagation

## Success Criteria
- Can write and read bytes to/from local filesystem
- Handles cancellation tokens correctly
- Creates directories as needed
- Provides clean abstraction for future cloud implementations
