using MLFramework.Core;

namespace MLFramework.Fusion;

/// <summary>
/// Maps tensor data types to CUDA type names
/// </summary>
public static class CudaTypeMapper
{
    public static string GetCudaTypeName(DataType dtype, int vectorWidth = 1)
    {
        var baseType = dtype switch
        {
            DataType.Float32 => "float",
            DataType.Float16 => "half",
            DataType.BFloat16 => "__nv_bfloat16",
            DataType.Int32 => "int",
            DataType.Int64 => "long long",
            DataType.Int16 => "short",
            DataType.Int8 => "char",
            DataType.UInt8 => "unsigned char",
            DataType.Bool => "bool",
            _ => throw new NotSupportedException($"Unsupported dtype: {dtype}")
        };

        if (vectorWidth > 1)
        {
            return GetVectorTypeName(baseType, vectorWidth);
        }

        return baseType;
    }

    private static string GetVectorTypeName(string baseType, int width)
    {
        return width switch
        {
            2 => $"{baseType}2",
            3 => $"{baseType}3",
            4 => $"{baseType}4",
            8 => baseType switch
            {
                "float" => "float4",
                "half" => "half4",
                _ => $"{baseType}8"
            },
            _ => baseType
        };
    }
}
