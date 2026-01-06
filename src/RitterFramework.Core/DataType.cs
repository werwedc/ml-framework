namespace RitterFramework.Core;

/// <summary>
/// Data type enumeration for tensor data
/// </summary>
public enum DataType
{
    // Existing types
    Float32 = 0,
    Float64 = 1,
    Int32 = 2,
    Int64 = 3,
    Int16 = 4,
    Int8 = 5,
    UInt8 = 6,
    Bool = 7,

    // AMP-specific types
    Float16 = 10,   // Half precision (IEEE 754)
    BFloat16 = 11   // Brain Float (Google's format)
}
