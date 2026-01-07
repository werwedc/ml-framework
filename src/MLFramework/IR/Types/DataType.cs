namespace MLFramework.IR.Types
{
    /// <summary>
    /// Represents the data type of tensor elements in the IR system.
    /// </summary>
    public enum DataType
    {
        /// <summary>32-bit floating point</summary>
        Float32,
        /// <summary>64-bit floating point</summary>
        Float64,
        /// <summary>16-bit floating point</summary>
        Float16,
        /// <summary>Brain floating point (16-bit)</summary>
        BFloat16,
        /// <summary>8-bit signed integer</summary>
        Int8,
        /// <summary>16-bit signed integer</summary>
        Int16,
        /// <summary>32-bit signed integer</summary>
        Int32,
        /// <summary>64-bit signed integer</summary>
        Int64,
        /// <summary>8-bit unsigned integer</summary>
        UInt8,
        /// <summary>16-bit unsigned integer</summary>
        UInt16,
        /// <summary>32-bit unsigned integer</summary>
        UInt32,
        /// <summary>64-bit unsigned integer</summary>
        UInt64,
        /// <summary>Boolean</summary>
        Bool
    }
}
