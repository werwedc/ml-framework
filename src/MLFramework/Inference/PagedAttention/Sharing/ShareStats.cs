namespace MlFramework.Inference.PagedAttention.Sharing;

/// <summary>
/// Statistics about block sharing.
/// </summary>
public class ShareStats
{
    public int TotalSharedBlocks { get; set; }
    public int TotalBlocksReferenced { get; set; }
    public int TotalReferences { get; set; }
    public double AverageReferenceCount { get; set; }

    public override string ToString()
    {
        return $"ShareStats: " +
               $"Shared={TotalSharedBlocks}, " +
               $"Referenced={TotalBlocksReferenced}, " +
               $"AvgRefCount={AverageReferenceCount:F2}";
    }
}
