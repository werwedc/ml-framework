namespace MlFramework.Inference.PagedAttention.Models;

/// <summary>
/// Result of a block allocation operation.
/// </summary>
public class BlockAllocationResult
{
    /// <summary>
    /// The allocated block ID.
    /// </summary>
    public int BlockId { get; set; }

    /// <summary>
    /// Flag indicating if allocation was successful.
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Error message if allocation failed.
    /// </summary>
    public string? ErrorMessage { get; set; }

    public static BlockAllocationResult Successful(int blockId)
    {
        return new BlockAllocationResult
        {
            BlockId = blockId,
            Success = true
        };
    }

    public static BlockAllocationResult Failed(string errorMessage)
    {
        return new BlockAllocationResult
        {
            Success = false,
            ErrorMessage = errorMessage
        };
    }
}
