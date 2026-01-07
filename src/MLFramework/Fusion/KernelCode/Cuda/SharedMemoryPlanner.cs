using MLFramework.Core;

namespace MLFramework.Fusion;

/// <summary>
/// Plans shared memory usage for tiling optimizations
/// </summary>
public class SharedMemoryPlanner : ISharedMemoryPlanner
{
    private const int MaxSharedMemory = 48 * 1024; // 48KB per SM

    public MemoryLayout PlanMemory(FusionIR ir, GenerationOptions options)
    {
        if (!options.EnableSharedMemory)
            return ir.MemoryLayout;

        // Calculate required shared memory for tiling
        var tilingConfig = ComputeTilingConfiguration(ir, options);

        if (tilingConfig == null)
            return ir.MemoryLayout;

        return ir.MemoryLayout with
        {
            SharedMemoryBytes = tilingConfig.TotalBytes,
            TensorLayout = OptimizeLayoutForTiling(ir, tilingConfig)
        };
    }

    private TilingConfiguration? ComputeTilingConfiguration(FusionIR ir, GenerationOptions options)
    {
        if (ir.Nodes.Count == 0)
            return null;

        var primaryOp = ir.Nodes[0].OriginalOpType;

        return primaryOp switch
        {
            "Conv2D" => ComputeConvTiling(ir, options),
            "Linear" => ComputeLinearTiling(ir, options),
            _ when IsElementWiseChain(ir) => ComputeElementWiseTiling(ir, options),
            _ => null
        };
    }

    private TilingConfiguration? ComputeConvTiling(FusionIR ir, GenerationOptions options)
    {
        // Tiling for convolution: tile input and weights into shared memory
        if (ir.Variables.Count == 0)
            return null;

        var inputShape = ir.Variables[0].Shape;
        var dtypeSize = GetDataTypeSize(ir.Variables[0].DataType);

        // Tile size (configurable based on kernel size)
        int tileH = 8;
        int tileW = 8;
        int tileC = 32;

        var inputTileSize = tileH * tileW * tileC * dtypeSize;
        var weightTileSize = tileH * tileW * tileC * dtypeSize;
        var totalBytes = inputTileSize + weightTileSize;

        if (totalBytes > MaxSharedMemory)
            return null;

        return new TilingConfiguration
        {
            TileHeight = tileH,
            TileWidth = tileW,
            TileChannels = tileC,
            TotalBytes = totalBytes
        };
    }

    private TilingConfiguration? ComputeLinearTiling(FusionIR ir, GenerationOptions options)
    {
        // Tiling for matrix multiplication
        if (ir.Variables.Count == 0)
            return null;

        var inputShape = ir.Variables[0].Shape;
        var dtypeSize = GetDataTypeSize(ir.Variables[0].DataType);

        // Standard matrix multiplication tiling
        int tileM = 64;
        int tileN = 64;
        int tileK = 8;

        var tileSize = (tileM * tileK + tileK * tileN) * dtypeSize;
        var totalBytes = tileSize;

        if (totalBytes > MaxSharedMemory)
            return null;

        return new TilingConfiguration
        {
            TileHeight = tileM,
            TileWidth = tileN,
            TileChannels = tileK,
            TotalBytes = totalBytes
        };
    }

    private TilingConfiguration? ComputeElementWiseTiling(FusionIR ir, GenerationOptions options)
    {
        // Element-wise operations don't need shared memory tiling
        return null;
    }

    private TensorLayout OptimizeLayoutForTiling(FusionIR ir, TilingConfiguration config)
    {
        // Use NCHW for better memory coalescing
        return TensorLayout.NCHW;
    }

    private int GetDataTypeSize(DataType dtype)
    {
        return dtype switch
        {
            DataType.Float32 or DataType.Int32 => 4,
            DataType.Float16 or DataType.Int16 => 2,
            DataType.Int8 => 1,
            DataType.Int64 => 8,
            _ => 4
        };
    }

    private bool IsElementWiseChain(FusionIR ir)
    {
        return ir.Nodes.All(n =>
            n.OriginalOpType is "Add" or "Sub" or "Mul" or "Div" or
            "ReLU" or "Sigmoid" or "Tanh" or "Exp" or "Log" or "Sqrt" or "Abs");
    }
}
