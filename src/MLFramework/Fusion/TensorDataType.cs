using MLFramework.Core;

namespace MLFramework.Fusion;

/// <summary>
/// Alias for DataType to maintain fusion-specific namespace
/// </summary>
public static class TensorDataType
{
    public const DataType Float32 = DataType.Float32;
    public const DataType Float16 = DataType.Float16;
    public const DataType BFloat16 = DataType.BFloat16;
    public const DataType Int32 = DataType.Int32;
    public const DataType Int64 = DataType.Int64;
    public const DataType Int16 = DataType.Int16;
    public const DataType Int8 = DataType.Int8;
    public const DataType UInt8 = DataType.UInt8;
    public const DataType Float64 = DataType.Float64;
}
