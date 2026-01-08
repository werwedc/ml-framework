using MachineLearning.Visualization.Events;

namespace MLFramework.Visualization.Storage;

/// <summary>
/// Writer for persisting events to disk in a file-based format
/// </summary>
internal class EventFileWriter : IDisposable
{
    private readonly FileStream _fileStream;
    private readonly BinaryWriter _binaryWriter;
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>
    /// Gets the current file path
    /// </summary>
    public string FilePath { get; }

    /// <summary>
    /// Gets the current size of the file in bytes
    /// </summary>
    public long FileSize => _fileStream?.Length ?? 0;

    /// <summary>
    /// Creates a new EventFileWriter
    /// </summary>
    /// <param name="filePath">Path to the file to write</param>
    /// <param name="append">Whether to append to existing file or create new</param>
    public EventFileWriter(string filePath, bool append = false)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));
        }

        FilePath = filePath;

        // Create directory if it doesn't exist
        var directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        // Open file stream
        _fileStream = new FileStream(
            filePath,
            append ? FileMode.Append : FileMode.Create,
            FileAccess.Write,
            FileShare.Read,
            bufferSize: 8192,
            options: FileOptions.WriteThrough);

        _binaryWriter = new BinaryWriter(_fileStream);
    }

    /// <summary>
    /// Writes an event to the file
    /// </summary>
    /// <param name="eventData">Event to write</param>
    public void WriteEvent(Event eventData)
    {
        if (eventData == null)
        {
            throw new ArgumentNullException(nameof(eventData));
        }

        lock (_lock)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(EventFileWriter));
            }

            // Serialize the event
            byte[] eventDataBytes = eventData.Serialize();

            // Write length prefix (4 bytes for length)
            _binaryWriter.Write(eventDataBytes.Length);

            // Write event data
            _binaryWriter.Write(eventDataBytes);
        }
    }

    /// <summary>
    /// Writes an event to the file asynchronously
    /// </summary>
    /// <param name="eventData">Event to write</param>
    public Task WriteEventAsync(Event eventData)
    {
        if (eventData == null)
        {
            throw new ArgumentNullException(nameof(eventData));
        }

        return Task.Run(() => WriteEvent(eventData));
    }

    /// <summary>
    /// Flushes any buffered data to disk
    /// </summary>
    public void Flush()
    {
        lock (_lock)
        {
            if (_disposed)
            {
                return;
            }

            _binaryWriter.Flush();
            _fileStream.Flush();
        }
    }

    /// <summary>
    /// Disposes of the writer and closes the file
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes of the writer and closes the file
    /// </summary>
    /// <param name="disposing">True if disposing managed resources</param>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _binaryWriter?.Dispose();
                _fileStream?.Dispose();
            }

            _disposed = true;
        }
    }

    /// <summary>
    /// Finalizer
    /// </summary>
    ~EventFileWriter()
    {
        Dispose(false);
    }
}
