using MachineLearning.Visualization.Events;
using System.Text.Json;

namespace MLFramework.Visualization.Storage;

/// <summary>
/// Reader for reading events from disk in a file-based format
/// </summary>
internal class EventFileReader : IDisposable
{
    private readonly FileStream _fileStream;
    private readonly BinaryReader _binaryReader;
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>
    /// Gets the file path being read
    /// </summary>
    public string FilePath { get; }

    /// <summary>
    /// Creates a new EventFileReader
    /// </summary>
    /// <param name="filePath">Path to the file to read</param>
    public EventFileReader(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"File not found: {filePath}", filePath);
        }

        FilePath = filePath;

        // Open file stream
        _fileStream = new FileStream(
            filePath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.ReadWrite,
            bufferSize: 8192,
            options: FileOptions.SequentialScan);

        _binaryReader = new BinaryReader(_fileStream);
    }

    /// <summary>
    /// Reads all events from the file
    /// </summary>
    /// <returns>Enumerable of events</returns>
    public IEnumerable<Event> ReadEvents()
    {
        lock (_lock)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(EventFileReader));
            }

            // Reset position to beginning of file
            _fileStream.Position = 0;

            var events = new List<Event>();

            while (_fileStream.Position < _fileStream.Length)
            {
                try
                {
                    // Read length prefix (4 bytes)
                    int length = _binaryReader.ReadInt32();

                    if (length <= 0 || length > _fileStream.Length - _fileStream.Position)
                    {
                        // Invalid length, skip to next event
                        break;
                    }

                    // Read event data
                    byte[] eventDataBytes = _binaryReader.ReadBytes(length);

                    // Deserialize event
                    Event? eventObj = DeserializeEvent(eventDataBytes);
                    if (eventObj != null)
                    {
                        events.Add(eventObj);
                    }
                }
                catch (EndOfStreamException)
                {
                    // Reached end of stream unexpectedly, stop reading
                    break;
                }
                catch (Exception ex)
                {
                    // Log error and continue with next event
                    // In production, you might want to log this
                    Console.WriteLine($"Error reading event: {ex.Message}");
                    break;
                }
            }

            return events;
        }
    }

    /// <summary>
    /// Reads all events from the file asynchronously
    /// </summary>
    /// <returns>Task with enumerable of events</returns>
    public async Task<IEnumerable<Event>> ReadEventsAsync()
    {
        return await Task.Run(() => ReadEvents()).ConfigureAwait(false);
    }

    /// <summary>
    /// Deserializes bytes to an event object
    /// </summary>
    /// <param name="data">Serialized event data</param>
    /// <returns>Deserialized event or null if deserialization fails</returns>
    private Event? DeserializeEvent(byte[] data)
    {
        try
        {
            // Parse the JSON to determine event type
            var json = JsonDocument.Parse(data);
            var root = json.RootElement;

            if (!root.TryGetProperty("type", out var typeProperty))
            {
                return null;
            }

            string eventType = typeProperty.GetString() ?? string.Empty;

            // Create appropriate event based on type
            Event? eventObj = eventType switch
            {
                "ScalarMetric" => new ScalarMetricEvent("", 0f),
                "Histogram" => new HistogramEvent("", Array.Empty<float>()),
                "ProfilingStart" => new ProfilingStartEvent(""),
                "ProfilingEnd" => new ProfilingEndEvent(""),
                "MemoryAllocation" => new MemoryAllocationEvent("", 0),
                "TensorOperation" => new TensorOperationEvent("", Array.Empty<int[]>(), Array.Empty<int>()),
                "ComputationalGraph" => new ComputationalGraphEvent("", ""),
                _ => null
            };

            if (eventObj != null)
            {
                eventObj.Deserialize(data);
                return eventObj;
            }

            return null;
        }
        catch
        {
            // Deserialization failed
            return null;
        }
    }

    /// <summary>
    /// Disposes of the reader and closes the file
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes of the reader and closes the file
    /// </summary>
    /// <param name="disposing">True if disposing managed resources</param>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _binaryReader?.Dispose();
                _fileStream?.Dispose();
            }

            _disposed = true;
        }
    }

    /// <summary>
    /// Finalizer
    /// </summary>
    ~EventFileReader()
    {
        Dispose(false);
    }
}
