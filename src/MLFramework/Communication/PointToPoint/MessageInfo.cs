namespace MLFramework.Communication.PointToPoint;

/// <summary>
/// Information about an incoming message
/// </summary>
public class MessageInfo
{
    /// <summary>
    /// Source rank of the message
    /// </summary>
    public int SourceRank { get; set; }

    /// <summary>
    /// Tag of the message
    /// </summary>
    public int Tag { get; set; }

    /// <summary>
    /// Number of elements in the message
    /// </summary>
    public long Count { get; set; }

    /// <summary>
    /// Data type of the message
    /// </summary>
    public Type? DataType { get; set; }

    /// <summary>
    /// Create a new MessageInfo instance
    /// </summary>
    public MessageInfo(int sourceRank, int tag, long count, Type dataType)
    {
        SourceRank = sourceRank;
        Tag = tag;
        Count = count;
        DataType = dataType;
    }

    /// <summary>
    /// Default constructor for serialization
    /// </summary>
    public MessageInfo()
    {
    }

    public override string ToString()
    {
        return $"MessageInfo(SourceRank={SourceRank}, Tag={Tag}, Count={Count}, DataType={DataType?.Name})";
    }
}
