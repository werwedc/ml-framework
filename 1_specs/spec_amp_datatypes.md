# Spec: AMP Datatypes (Half and BFloat16)

## Overview
Define the fundamental low-precision numeric types (Half and BFloat16) that enable Automatic Mixed Precision training in the ML framework.

## Class Specification

### 1. Half Struct (16-bit Float)

**File:** `src/MLFramework/Amp/Half.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// 16-bit floating point type (IEEE 754 half-precision)
    /// Sign: 1 bit, Exponent: 5 bits, Mantissa: 10 bits
    /// Range: ~6e-5 to 65504, ~3 decimal digits of precision
    /// </summary>
    public readonly struct Half : IComparable<Half>, IEquatable<Half>
    {
        private readonly ushort _value;

        // Constructors
        public Half(float value);
        public Half(double value);
        public Half(ushort rawBits);

        // Conversion operators
        public static explicit operator float(Half h);
        public static explicit operator double(Half h);
        public static explicit operator Half(float f);
        public static explicit operator Half(double d);

        // Arithmetic operators
        public static Half operator +(Half left, Half right);
        public static Half operator -(Half left, Half right);
        public static Half operator *(Half left, Half right);
        public static Half operator /(Half left, Half right);

        // Comparison operators
        public static bool operator ==(Half left, Half right);
        public static bool operator !=(Half left, Half right);
        public static bool operator <(Half left, Half right);
        public static bool operator >(Half left, Half right);
        public static bool operator <=(Half left, Half right);
        public static bool operator >=(Half left, Half right);

        // Interface implementations
        public int CompareTo(Half other);
        public bool Equals(Half other);
        public override bool Equals(object obj);
        public override int GetHashCode();

        // Properties
        public bool IsNaN { get; }
        public bool IsInfinity { get; }
        public bool IsPositiveInfinity { get; }
        public bool IsNegativeInfinity { get; }

        // Constants
        public static readonly Half Epsilon = new Half(0x0001);
        public static readonly Half MaxValue = new Half(0x7BFF);
        public static readonly Half MinValue = new Half(0xFBFF);
        public static readonly Half PositiveInfinity = new Half(0x7C00);
        public static readonly Half NegativeInfinity = new Half(0xFC00);
        public static readonly Half NaN = new Half(0x7E00);
    }
}
```

### 2. BFloat16 Struct (16-bit Brain Float)

**File:** `src/MLFramework/Amp/BFloat16.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// 16-bit Brain Floating Point type (Google's format)
    /// Sign: 1 bit, Exponent: 8 bits, Mantissa: 7 bits
    /// Range: Same as FP32 (~3e-38 to 3e38), ~2 decimal digits of precision
    /// Better dynamic range than FP16, preferred for Transformers
    /// </summary>
    public readonly struct BFloat16 : IComparable<BFloat16>, IEquatable<BFloat16>
    {
        private readonly ushort _value;

        // Constructors
        public BFloat16(float value);
        public BFloat16(double value);
        public BFloat16(ushort rawBits);

        // Conversion operators
        public static explicit operator float(BFloat16 bf);
        public static explicit operator double(BFloat16 bf);
        public static explicit operator BFloat16(float f);
        public static explicit operator BFloat16(double d);

        // Arithmetic operators
        public static BFloat16 operator +(BFloat16 left, BFloat16 right);
        public static BFloat16 operator -(BFloat16 left, BFloat16 right);
        public static BFloat16 operator *(BFloat16 left, BFloat16 right);
        public static BFloat16 operator /(BFloat16 left, BFloat16 right);

        // Comparison operators
        public static bool operator ==(BFloat16 left, BFloat16 right);
        public static bool operator !=(BFloat16 left, BFloat16 right);
        public static bool operator <(BFloat16 left, BFloat16 right);
        public static bool operator >(BFloat16 left, BFloat16 right);
        public static bool operator <=(BFloat16 left, BFloat16 right);
        public static bool operator >=(BFloat16 left, BFloat16 right);

        // Interface implementations
        public int CompareTo(BFloat16 other);
        public bool Equals(BFloat16 other);
        public override bool Equals(object obj);
        public override int GetHashCode();

        // Properties
        public bool IsNaN { get; }
        public bool IsInfinity { get; }
        public bool IsPositiveInfinity { get; }
        public bool IsNegativeInfinity { get; }

        // Constants
        public static readonly BFloat16 Epsilon = new BFloat16(0x007F);
        public static readonly BFloat16 MaxValue = new BFloat16(0x7F7F);
        public static readonly BFloat16 MinValue = new BFloat16(0xFF7F);
        public static readonly BFloat16 PositiveInfinity = new BFloat16(0x7F80);
        public static readonly BFloat16 NegativeInfinity = new BFloat16(0xFF80);
        public static readonly BFloat16 NaN = new BFloat16(0x7FC0);
    }
}
```

### 3. Casting Utilities

**File:** `src/MLFramework/Amp/AmpCast.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// High-performance casting utilities for AMP
    /// </summary>
    public static class AmpCast
    {
        /// <summary>
        /// Cast float array to Half (zero-copy when possible)
        /// </summary>
        public static Half[] CastToHalf(float[] input);

        /// <summary>
        /// Cast float array to BFloat16 (zero-copy when possible)
        /// </summary>
        public static BFloat16[] CastToBFloat16(float[] input);

        /// <summary>
        /// Cast Half array to float
        /// </summary>
        public static float[] CastToFloat(Half[] input);

        /// <summary>
        /// Cast BFloat16 array to float
        /// </summary>
        public static float[] CastToFloat(BFloat16[] input);

        /// <summary>
        /// In-place cast Half to float
        /// </summary>
        public static void CastInPlace(Half[] input, float[] output);

        /// <summary>
        /// In-place cast BFloat16 to float
        /// </summary>
        public static void CastInPlace(BFloat16[] input, float[] output);
    }
}
```

## Implementation Notes

### Half Implementation Details
- Use bit manipulation for efficient float <-> Half conversion
- Handle special cases: NaN, Infinity, Denormals
- Consider using unsafe code with pointers for performance

### BFloat16 Implementation Details
- Truncate mantissa from FP32 (easier conversion than Half)
- No need for denormal handling (same exponent range as FP32)
- Use bitwise operations for fast conversion

### Performance Considerations
- Use SIMD (System.Numerics) for bulk operations if possible
- Cache conversion factors to avoid repeated computation
- Consider lazy evaluation for large arrays

## Dependencies
- System (primitive types)
- System.Numerics (for potential SIMD optimization)

## Testing Requirements
- Test all arithmetic operations for accuracy
- Verify special value handling (NaN, Infinity)
- Test boundary conditions (MaxValue, MinValue, Epsilon)
- Benchmark conversion performance
- Test rounding behavior

## Success Criteria
- [ ] All arithmetic operations produce results within 0.1% of FP32
- [ ] Special values (NaN, Infinity) handled correctly
- [ ] Conversion overhead < 10% of operation time
- [ ] Zero-copy casting works when possible
- [ ] All unit tests pass
