using System.Diagnostics;
using System.Text;
using MachineLearning.Visualization.Events;

namespace MachineLearning.Visualization.Memory;

/// <summary>
/// Represents a memory-related event for tracking allocations, deallocations, and usage
/// </summary>
public class MemoryEvent : Event
{
    /// <summary>
    /// Gets the type of memory event
    /// </summary>
    public MemoryEventType MemoryEventType { get; }

    /// <summary>
    /// Gets the memory address (for tracking allocations)
    /// </summary>
    public long Address { get; }

    /// <summary>
    /// Gets the size in bytes
    /// </summary>
    public long SizeBytes { get; }

    /// <summary>
    /// Gets the total allocated bytes at the time of this event
    /// </summary>
    public long TotalAllocatedBytes { get; }

    /// <summary>
    /// Gets the total freed bytes at the time of this event
    /// </summary>
    public long TotalFreeBytes { get; }

    /// <summary>
    /// Gets the allocation type (e.g., "GPU", "CPU", "Pinned")
    /// </summary>
    public string AllocationType { get; }

    /// <summary>
    /// Gets the stack trace at allocation time (optional, for debugging)
    /// </summary>
    public StackTrace? AllocationStackTrace { get; }

    /// <summary>
    /// Gets the GC generation (for GC events)
    /// </summary>
    public int? GCGeneration { get; }

    /// <summary>
    /// Private constructor for deserialization
    /// </summary>
    private MemoryEvent()
    {
    }

    /// <summary>
    /// Creates a new memory event
    /// </summary>
    public MemoryEvent(
        MemoryEventType memoryEventType,
        long address,
        long sizeBytes,
        long totalAllocatedBytes,
        long totalFreeBytes,
        string allocationType,
        StackTrace? allocationStackTrace = null,
        int? gcGeneration = null)
    {
        if (sizeBytes < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeBytes), "Size bytes cannot be negative");
        }

        if (totalAllocatedBytes < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(totalAllocatedBytes), "Total allocated bytes cannot be negative");
        }

        if (totalFreeBytes < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(totalFreeBytes), "Total free bytes cannot be negative");
        }

        if (string.IsNullOrEmpty(allocationType))
        {
            throw new ArgumentException("Allocation type cannot be null or empty", nameof(allocationType));
        }

        MemoryEventType = memoryEventType;
        Address = address;
        SizeBytes = sizeBytes;
        TotalAllocatedBytes = totalAllocatedBytes;
        TotalFreeBytes = totalFreeBytes;
        AllocationType = allocationType;
        AllocationStackTrace = allocationStackTrace;
        GCGeneration = gcGeneration;
    }

    /// <summary>
    /// Serializes the event to bytes for storage
    /// </summary>
    public override byte[] Serialize()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"{Timestamp:o}");
        sb.AppendLine($"{EventId}");
        sb.AppendLine($"{(int)MemoryEventType}");
        sb.AppendLine($"{Address}");
        sb.AppendLine($"{SizeBytes}");
        sb.AppendLine($"{TotalAllocatedBytes}");
        sb.AppendLine($"{TotalFreeBytes}");
        sb.AppendLine($"{AllocationType}");
        sb.AppendLine($"{AllocationStackTrace != null}");
        sb.AppendLine($"{GCGeneration.HasValue}");
        if (GCGeneration.HasValue)
        {
            sb.AppendLine($"{GCGeneration.Value}");
        }

        return Encoding.UTF8.GetBytes(sb.ToString());
    }

    /// <summary>
    /// Deserializes bytes back to an event
    /// </summary>
    public override void Deserialize(byte[] data)
    {
        if (data == null || data.Length == 0)
        {
            throw new ArgumentException("Data cannot be null or empty", nameof(data));
        }

        var lines = Encoding.UTF8.GetString(data).Split(new[] { Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries);
        if (lines.Length < 11)
        {
            throw new ArgumentException("Invalid data format", nameof(data));
        }

        var lineIndex = 0;
        Timestamp = DateTime.ParseExact(lines[lineIndex++], "o", null);
        var eventId = Guid.Parse(lines[lineIndex++]);
        MemoryEventType = (MemoryEventType)int.Parse(lines[lineIndex++]);
        Address = long.Parse(lines[lineIndex++]);
        SizeBytes = long.Parse(lines[lineIndex++]);
        TotalAllocatedBytes = long.Parse(lines[lineIndex++]);
        TotalFreeBytes = long.Parse(lines[lineIndex++]);
        AllocationType = lines[lineIndex++];
        var hasStackTrace = bool.Parse(lines[lineIndex++]);
        var hasGCGeneration = bool.Parse(lines[lineIndex++]);

        AllocationStackTrace = hasStackTrace ? new StackTrace() : null;
        GCGeneration = hasGCGeneration ? int.Parse(lines[lineIndex++]) : null;

        // Set the EventId using reflection since it's read-only
        var eventIdProperty = typeof(Event).GetProperty(nameof(EventId));
        eventIdProperty?.SetValue(this, eventId);
    }
}
