# Spec: AMP DataType Extensions

## Overview
Extend the existing DataType enum to support FP16 and BF16 precision modes for Automatic Mixed Precision.

## Class Specification

### 1. DataType Enum Extensions

**File:** `src/MLFramework/Core/DataType.cs` (extend existing file)

```csharp
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
        public static int GetSize(this DataType dtype);

        /// <summary>
        /// Checks if the type is floating point
        /// </summary>
        public static bool IsFloatType(this DataType dtype);

        /// <summary>
        /// Checks if the type is low precision (FP16/BF16)
        /// </summary>
        public static bool IsLowPrecision(this DataType dtype);

        /// <summary>
        /// Gets the default higher precision type for casting
        /// FP16 -> Float32, BF16 -> Float32
        /// </summary>
        public static DataType GetHigherPrecision(this DataType dtype);

        /// <summary>
        /// Gets the default lower precision type for AMP
        /// Float32 -> BF16 (or Float16 based on preference)
        /// </summary>
        public static DataType GetLowerPrecision(this DataType dtype);
    }
}
```

### 2. DataTypeInfo Class

**File:** `src/MLFramework/Core/DataTypeInfo.cs`

```csharp
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
        public static int SizeOf(DataType dtype);

        /// <summary>
        /// Gets the name of the data type
        /// </summary>
        public static string GetName(DataType dtype);

        /// <summary>
        /// Gets the type code for the data type
        /// </summary>
        public static TypeCode GetTypeCode(DataType dtype);

        /// <summary>
        /// Gets the maximum representable value for the type
        /// </summary>
        public static double GetMaxValue(DataType dtype);

        /// <summary>
        /// Gets the minimum representable value for the type
        /// </summary>
        public static double GetMinValue(DataType dtype);

        /// <summary>
        /// Gets the epsilon (smallest positive number) for the type
        /// </summary>
        public static double GetEpsilon(DataType dtype);

        /// <summary>
        /// Checks if the type supports NaN
        /// </summary>
        public static bool SupportsNaN(DataType dtype);

        /// <summary>
        /// Checks if the type supports Infinity
        /// </summary>
        public static bool SupportsInfinity(DataType dtype);

        /// <summary>
        /// Gets the precision (number of significant decimal digits)
        /// </summary>
        public static int GetPrecision(DataType dtype);

        /// <summary>
        /// Gets the dynamic range (log10 of max/min ratio)
        /// </summary>
        public static double GetDynamicRange(DataType dtype);
    }
}
```

### 3. AmpConfig Class

**File:** `src/MLFramework/Amp/AmpConfig.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Configuration settings for Automatic Mixed Precision
    /// </summary>
    public class AmpConfig
    {
        /// <summary>
        /// Gets or sets the target precision for AMP (FP16 or BF16)
        /// </summary>
        public DataType TargetPrecision { get; set; }

        /// <summary>
        /// Gets or sets whether to enable AMP globally
        /// </summary>
        public bool Enabled { get; set; }

        /// <summary>
        /// Gets or sets the default higher precision type (usually Float32)
        /// </summary>
        public DataType HigherPrecision { get; set; }

        /// <summary>
        /// Gets or sets whether to use view casting (zero-copy) when possible
        /// </summary>
        public bool UseViewCasting { get; set; }

        /// <summary>
        /// Gets or sets whether to enable kernel fusion across precision boundaries
        /// </summary>
        public bool EnableKernelFusion { get; set; }

        /// <summary>
        /// Creates a default AMP config with BF16 (recommended for Transformers)
        /// </summary>
        public static AmpConfig CreateDefault();

        /// <summary>
        /// Creates an AMP config with FP16 (better for older hardware)
        /// </summary>
        public static AmpConfig CreateFp16();

        /// <summary>
        /// Creates an AMP config with BF16 (better for dynamic range)
        /// </summary>
        public static AmpConfig CreateBf16();
    }
}
```

## Implementation Details

### DataType Extension Methods
- **GetSize()**: Returns 2 for FP16/BF16, 4 for FP32, 8 for FP64, etc.
- **IsFloatType()**: Returns true for FP16, BF16, FP32, FP64
- **IsLowPrecision()**: Returns true for FP16 and BF16
- **GetHigherPrecision()**: Returns Float32 for FP16/BF16
- **GetLowerPrecision()**: Returns BF16 for Float32 (configurable)

### DataTypeInfo Constants
- FP16: 2 bytes, 3 decimal digits, ~4.9e-4 dynamic range
- BF16: 2 bytes, 2 decimal digits, ~38 decimal digits range (same as FP32)
- Float32: 4 bytes, 7 decimal digits, ~38 decimal digits range

### AmpConfig Defaults
```csharp
TargetPrecision = DataType.BFloat16
Enabled = true
HigherPrecision = DataType.Float32
UseViewCasting = true
EnableKernelFusion = true
```

## Dependencies
- Existing DataType enum in MLFramework.Core
- System (for TypeCode)

## Testing Requirements
- Test all extension methods for all DataType values
- Verify GetSize() returns correct values
- Test precision hierarchy (FP16 -> FP32 -> FP64)
- Test AmpConfig creation methods
- Verify dynamic range calculations

## Success Criteria
- [ ] All DataType enum values are accessible
- [ ] Extension methods work correctly for all types
- [ ] DataTypeInfo returns accurate metadata
- [ ] AmpConfig creates valid configurations
- [ ] Unit tests cover all edge cases
- [ ] No breaking changes to existing DataType usage
