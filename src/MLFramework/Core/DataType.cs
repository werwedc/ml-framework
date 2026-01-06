namespace MLFramework.Core
{
    /// <summary>
    /// Data type enumeration with AMP support
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

    /// <summary>
    /// Extension methods for DataType
    /// </summary>
    public static class DataTypeExtensions
    {
        /// <summary>
        /// Gets the byte size of the data type
        /// </summary>
        public static int GetSize(this DataType dtype)
        {
            return dtype switch
            {
                DataType.Float16 => 2,
                DataType.BFloat16 => 2,
                DataType.Float32 => 4,
                DataType.Float64 => 8,
                DataType.Int32 => 4,
                DataType.Int64 => 8,
                DataType.Int16 => 2,
                DataType.Int8 => 1,
                DataType.UInt8 => 1,
                DataType.Bool => 1,
                _ => throw new ArgumentException($"Unknown data type: {dtype}")
            };
        }

        /// <summary>
        /// Checks if the type is floating point
        /// </summary>
        public static bool IsFloatType(this DataType dtype)
        {
            return dtype switch
            {
                DataType.Float16 => true,
                DataType.BFloat16 => true,
                DataType.Float32 => true,
                DataType.Float64 => true,
                _ => false
            };
        }

        /// <summary>
        /// Checks if the type is low precision (FP16/BF16)
        /// </summary>
        public static bool IsLowPrecision(this DataType dtype)
        {
            return dtype == DataType.Float16 || dtype == DataType.BFloat16;
        }

        /// <summary>
        /// Gets the default higher precision type for casting
        /// FP16 -> Float32, BF16 -> Float32
        /// </summary>
        public static DataType GetHigherPrecision(this DataType dtype)
        {
            if (dtype.IsLowPrecision())
            {
                return DataType.Float32;
            }

            return dtype switch
            {
                DataType.Float32 => DataType.Float64,
                _ => dtype
            };
        }

        /// <summary>
        /// Gets the default lower precision type for AMP
        /// Float32 -> BF16 (or Float16 based on preference)
        /// </summary>
        public static DataType GetLowerPrecision(this DataType dtype)
        {
            return dtype switch
            {
                DataType.Float32 => DataType.BFloat16, // Default to BF16
                DataType.Float64 => DataType.Float32,
                _ => dtype
            };
        }
    }
}
