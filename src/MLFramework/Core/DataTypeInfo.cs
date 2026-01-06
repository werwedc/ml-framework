using System;

namespace MLFramework.Core
{
    /// <summary>
    /// Provides runtime information about data types
    /// </summary>
    public static class DataTypeInfo
    {
        /// <summary>
        /// Gets the byte size of a data type
        /// </summary>
        public static int SizeOf(DataType dtype)
        {
            return dtype.GetSize();
        }

        /// <summary>
        /// Gets the name of the data type
        /// </summary>
        public static string GetName(DataType dtype)
        {
            return dtype switch
            {
                DataType.Float16 => "Float16",
                DataType.BFloat16 => "BFloat16",
                DataType.Float32 => "Float32",
                DataType.Float64 => "Float64",
                DataType.Int32 => "Int32",
                DataType.Int64 => "Int64",
                DataType.Int16 => "Int16",
                DataType.Int8 => "Int8",
                DataType.UInt8 => "UInt8",
                DataType.Bool => "Bool",
                _ => "Unknown"
            };
        }

        /// <summary>
        /// Gets the type code for the data type
        /// </summary>
        public static TypeCode GetTypeCode(DataType dtype)
        {
            return dtype switch
            {
                DataType.Float16 => TypeCode.Single,
                DataType.BFloat16 => TypeCode.Single,
                DataType.Float32 => TypeCode.Single,
                DataType.Float64 => TypeCode.Double,
                DataType.Int32 => TypeCode.Int32,
                DataType.Int64 => TypeCode.Int64,
                DataType.Int16 => TypeCode.Int16,
                DataType.Int8 => TypeCode.SByte,
                DataType.UInt8 => TypeCode.Byte,
                DataType.Bool => TypeCode.Boolean,
                _ => TypeCode.Object
            };
        }

        /// <summary>
        /// Gets the maximum representable value for the type
        /// </summary>
        public static double GetMaxValue(DataType dtype)
        {
            return dtype switch
            {
                DataType.Float16 => 65504.0,
                DataType.BFloat16 => float.MaxValue,
                DataType.Float32 => float.MaxValue,
                DataType.Float64 => double.MaxValue,
                DataType.Int32 => int.MaxValue,
                DataType.Int64 => long.MaxValue,
                DataType.Int16 => short.MaxValue,
                DataType.Int8 => sbyte.MaxValue,
                DataType.UInt8 => byte.MaxValue,
                DataType.Bool => 1.0,
                _ => throw new ArgumentException($"Unknown data type: {dtype}")
            };
        }

        /// <summary>
        /// Gets the minimum representable value for the type
        /// </summary>
        public static double GetMinValue(DataType dtype)
        {
            return dtype switch
            {
                DataType.Float16 => -65504.0,
                DataType.BFloat16 => float.MinValue,
                DataType.Float32 => float.MinValue,
                DataType.Float64 => double.MinValue,
                DataType.Int32 => int.MinValue,
                DataType.Int64 => long.MinValue,
                DataType.Int16 => short.MinValue,
                DataType.Int8 => sbyte.MinValue,
                DataType.UInt8 => byte.MinValue,
                DataType.Bool => 0.0,
                _ => throw new ArgumentException($"Unknown data type: {dtype}")
            };
        }

        /// <summary>
        /// Gets the epsilon (smallest positive number) for the type
        /// </summary>
        public static double GetEpsilon(DataType dtype)
        {
            return dtype switch
            {
                DataType.Float16 => 0.00097656,  // 2^-10
                DataType.BFloat16 => 0.0078125,  // 2^-7
                DataType.Float32 => float.Epsilon,
                DataType.Float64 => double.Epsilon,
                DataType.Int32 => 1.0,
                DataType.Int64 => 1.0,
                DataType.Int16 => 1.0,
                DataType.Int8 => 1.0,
                DataType.UInt8 => 1.0,
                DataType.Bool => 1.0,
                _ => throw new ArgumentException($"Unknown data type: {dtype}")
            };
        }

        /// <summary>
        /// Checks if the type supports NaN
        /// </summary>
        public static bool SupportsNaN(DataType dtype)
        {
            return dtype.IsFloatType();
        }

        /// <summary>
        /// Checks if the type supports Infinity
        /// </summary>
        public static bool SupportsInfinity(DataType dtype)
        {
            return dtype.IsFloatType();
        }

        /// <summary>
        /// Gets the precision (number of significant decimal digits)
        /// </summary>
        public static int GetPrecision(DataType dtype)
        {
            return dtype switch
            {
                DataType.Float16 => 3,
                DataType.BFloat16 => 2,
                DataType.Float32 => 7,
                DataType.Float64 => 15,
                DataType.Int32 => 10,
                DataType.Int64 => 19,
                DataType.Int16 => 5,
                DataType.Int8 => 3,
                DataType.UInt8 => 3,
                DataType.Bool => 1,
                _ => throw new ArgumentException($"Unknown data type: {dtype}")
            };
        }

        /// <summary>
        /// Gets the dynamic range (log10 of max/min ratio)
        /// </summary>
        public static double GetDynamicRange(DataType dtype)
        {
            if (!dtype.IsFloatType())
            {
                return Math.Log10(Math.Abs(GetMaxValue(dtype) / GetMinValue(dtype)));
            }

            return dtype switch
            {
                DataType.Float16 => 4.9,
                DataType.BFloat16 => 38.0,
                DataType.Float32 => 38.0,
                DataType.Float64 => 308.0,
                _ => throw new ArgumentException($"Unknown data type: {dtype}")
            };
        }
    }
}
