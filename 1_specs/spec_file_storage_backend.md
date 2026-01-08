# Spec: File-Based Storage Backend

## Overview
Implement a file-based storage backend that persists events to disk, compatible with TensorBoard's event file format for ecosystem integration.

## Objectives
- Persist events to disk in TensorBoard-compatible format
- Support multiple event files with automatic rotation
- Enable efficient writes with buffering
- Allow reading events back for analysis

## API Design

```csharp
// TensorBoard event file format constants
public static class TensorBoardFormat
{
    public const string FileExtension = ".tfevents";
    public const string HeaderSize = "12";
}

// File-based storage backend
public class FileStorageBackend : StorageBackendBase
{
    public FileStorageBackend(StorageConfiguration config);
    public string LogDirectory { get; }
    public string CurrentFile { get; }
    public long MaxFileSize { get; set; } = 10 * 1024 * 1024; // 10MB
}

// Event file writer
internal class EventFileWriter : IDisposable
{
    public EventFileWriter(string filePath, bool append = false);
    public void WriteEvent(Event eventData);
    public Task WriteEventAsync(EventEventData);
    public void Flush();
    public long FileSize { get; }
}

// Event file reader (for debugging/analysis)
internal class EventFileReader : IDisposable
{
    public EventFileReader(string filePath);
    public IEnumerable<Event> ReadEvents();
    public Task<IEnumerable<Event>> ReadEventsAsync();
}
```

## Implementation Requirements

### 1. Event Serialization (30-45 min)
- Implement protobuf serialization for TensorBoard format:
  - Create protobuf message definitions or use existing library
  - Serialize `ScalarMetricEvent` to TensorBoard scalar format
  - Serialize `HistogramEvent` to TensorBoard histogram format
  - Serialize `ProfilingEvent` to TensorBoard profiling data
- Use `System.IO.Compression` for efficient storage
- Add CRC32 checksums for data integrity

### 2. EventFileWriter (45-60 min)
- Implement write with file handle management
- Support file append mode for continuing existing logs
- Track file size and report `FileSize` property
- Implement flush to ensure data is written to disk
- Use `FileStream` with appropriate buffering
- Handle file system errors gracefully
- Ensure thread-safe writes if used concurrently

### 3. EventFileReader (30-45 min)
- Read events from TensorBoard format files
- Parse protobuf messages
- Reconstruct C# Event objects
- Support async reading for large files
- Handle corrupt data gracefully (skip or report)

### 4. FileStorageBackend (45-60 min)
- Inherit from `StorageBackendBase`
- Implement directory creation if it doesn't exist
- Generate unique file names with timestamps:
  - Format: `events.out.tfevents.{timestamp}.{hostname}`
- Implement automatic file rotation when `MaxFileSize` is reached
- Store events using `EventFileWriter`
- Implement `GetEvents` by reading from multiple files in directory
- Handle file system permissions and errors
- Implement proper disposal of file handles

## File Structure
```
src/
  MLFramework.Visualization/
    Storage/
      FileStorageBackend.cs
      EventFileWriter.cs
      EventFileReader.cs
      Protobuf/
        EventSerializer.cs
        TensorBoardMessages.proto (or generated C# classes)

tests/
  MLFramework.Visualization.Tests/
    Storage/
      FileStorageBackendTests.cs
      EventFileWriterTests.cs
      EventFileReaderTests.cs
      EventSerializerTests.cs
```

## Dependencies
- `MLFramework.Visualization.Storage` (StorageBackendBase)
- `MLFramework.Visualization.Events` (Event types)
- Google.Protobuf (NuGet package)
- System.IO.Compression

## Integration Points
- Used by Visualizer main API when file storage is configured
- Outputs can be opened in TensorBoard for visualization
- Compatible with TensorBoard ecosystem tools

## Success Criteria
- Events are written in valid TensorBoard format
- Files can be opened in TensorBoard and display correctly
- File rotation works correctly and data is not lost
- Writing 10,000 events completes in <1 second
- Reading back events returns identical data
- Unit tests cover file rotation, serialization, and error cases
