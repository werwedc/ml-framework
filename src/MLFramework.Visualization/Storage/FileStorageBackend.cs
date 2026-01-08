using MachineLearning.Visualization.Events;
using MachineLearning.Visualization.Storage;

namespace MLFramework.Visualization.Storage;

/// <summary>
/// File-based storage backend that persists events to disk
/// </summary>
public class FileStorageBackend : StorageBackendBase
{
    private readonly object _lock = new();
    private EventFileWriter? _currentWriter;
    private long _currentFileSize;
    private long _currentStep = 0;

    /// <summary>
    /// Gets the log directory where events are stored
    /// </summary>
    public string LogDirectory { get; }

    /// <summary>
    /// Gets the current file being written to
    /// </summary>
    public string CurrentFile => _currentWriter?.FilePath ?? string.Empty;

    /// <summary>
    /// Gets or sets the maximum file size before rotation (default: 10MB)
    /// </summary>
    public long MaxFileSize { get; set; } = 10 * 1024 * 1024;

    /// <summary>
    /// Creates a new FileStorageBackend
    /// </summary>
    /// <param name="config">Storage configuration</param>
    public FileStorageBackend(StorageConfiguration config) : base(config)
    {
        if (config == null)
        {
            throw new ArgumentNullException(nameof(config));
        }

        // Use the connection string as the log directory
        LogDirectory = config.ConnectionString;

        // Ensure log directory exists
        if (!Directory.Exists(LogDirectory))
        {
            Directory.CreateDirectory(LogDirectory);
        }
    }

    /// <summary>
    /// Initializes the file storage backend
    /// </summary>
    /// <param name="connectionString">Log directory path</param>
    protected override void InitializeCore(string connectionString)
    {
        // The directory is already created in the constructor
        // Initialize the first event file writer
        CreateNewEventFile();
    }

    /// <summary>
    /// Shuts down the file storage backend
    /// </summary>
    protected override void ShutdownCore()
    {
        lock (_lock)
        {
            // Close current writer
            _currentWriter?.Dispose();
            _currentWriter = null;
            _currentFileSize = 0;
        }
    }

    /// <summary>
    /// Flushes events to the file
    /// </summary>
    /// <param name="events">Events to flush</param>
    protected override void FlushCore(IEnumerable<Event> events)
    {
        if (events == null)
        {
            return;
        }

        lock (_lock)
        {
            foreach (var eventData in events)
            {
                if (eventData == null)
                {
                    continue;
                }

                // Check if we need to rotate the file
                if (_currentFileSize >= MaxFileSize || _currentWriter == null)
                {
                    CreateNewEventFile();
                }

                // Write the event
                _currentWriter!.WriteEvent(eventData);
                _currentFileSize = _currentWriter.FileSize;
            }

            // Flush the writer to ensure data is written to disk
            _currentWriter?.Flush();
        }
    }

    /// <summary>
    /// Clears all events by deleting all event files
    /// </summary>
    protected override void ClearCore()
    {
        lock (_lock)
        {
            // Close current writer
            _currentWriter?.Dispose();
            _currentWriter = null;
            _currentFileSize = 0;

            // Delete all event files in the log directory
            if (Directory.Exists(LogDirectory))
            {
                var eventFiles = Directory.GetFiles(LogDirectory, "*.events");
                foreach (var file in eventFiles)
                {
                    try
                    {
                        File.Delete(file);
                    }
                    catch (Exception ex)
                    {
                        // Log error but continue with other files
                        Console.WriteLine($"Error deleting file {file}: {ex.Message}");
                    }
                }
            }

            // Create a new event file
            CreateNewEventFile();
        }
    }

    /// <summary>
    /// Gets events within a step range
    /// </summary>
    /// <param name="startStep">Starting step (inclusive)</param>
    /// <param name="endStep">Ending step (inclusive)</param>
    public override IEnumerable<Event> GetEvents(long startStep, long endStep)
    {
        var events = new List<Event>();

        if (!Directory.Exists(LogDirectory))
        {
            return events;
        }

        // Read from all event files in the directory
        var eventFiles = Directory.GetFiles(LogDirectory, "*.events")
            .OrderBy(f => f);

        foreach (var file in eventFiles)
        {
            try
            {
                using var reader = new EventFileReader(file);
                var fileEvents = reader.ReadEvents();

                // Filter events by step range
                foreach (var eventData in fileEvents)
                {
                    long step = ExtractStepFromEvent(eventData);
                    if (step >= startStep && step <= endStep)
                    {
                        events.Add(eventData);
                    }
                }
            }
            catch (Exception ex)
            {
                // Log error but continue with other files
                Console.WriteLine($"Error reading file {file}: {ex.Message}");
            }
        }

        return events.OrderBy(e => ExtractStepFromEvent(e));
    }

    /// <summary>
    /// Creates a new event file with a unique name
    /// </summary>
    private void CreateNewEventFile()
    {
        // Close current writer if it exists
        _currentWriter?.Dispose();

        // Generate unique file name with timestamp and hostname
        string timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString();
        string hostname = Environment.MachineName.Replace(" ", "_");
        string fileName = $"events.{timestamp}.{hostname}.events";
        string filePath = Path.Combine(LogDirectory, fileName);

        // Create new writer
        _currentWriter = new EventFileWriter(filePath, append: false);
        _currentFileSize = 0;
    }

    /// <summary>
    /// Extracts the step number from an event
    /// </summary>
    private static long ExtractStepFromEvent(Event eventData)
    {
        return eventData switch
        {
            ScalarMetricEvent sme => sme.Step,
            HistogramEvent he => he.Step,
            ProfilingStartEvent pse => pse.Step,
            ProfilingEndEvent pee => pee.Step,
            MemoryAllocationEvent mae => mae.Step,
            TensorOperationEvent toe => toe.Step,
            ComputationalGraphEvent cge => cge.Step,
            _ => -1
        };
    }
}
