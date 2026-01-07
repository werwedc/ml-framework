namespace MLFramework.Fusion;

/// <summary>
/// Record representing tiling configuration for shared memory
/// </summary>
public record TilingConfiguration
{
    public required int TileHeight { get; init; }
    public required int TileWidth { get; init; }
    public required int TileChannels { get; init; }
    public required int TotalBytes { get; init; }
}
