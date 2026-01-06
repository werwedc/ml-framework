using System;

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
        public BFloat16(float value) : this(ConvertToBFloat16(value)) { }

        public BFloat16(double value) : this(ConvertToBFloat16((float)value)) { }

        public BFloat16(ushort rawBits)
        {
            _value = rawBits;
        }

        // Conversion operators
        public static explicit operator float(BFloat16 bf) => ConvertToFloat(bf._value);

        public static explicit operator double(BFloat16 bf) => ConvertToFloat(bf._value);

        public static explicit operator BFloat16(float f) => new BFloat16(f);

        public static explicit operator BFloat16(double d) => new BFloat16((float)d);

        // Arithmetic operators
        public static BFloat16 operator +(BFloat16 left, BFloat16 right) =>
            new BFloat16((float)left + (float)right);

        public static BFloat16 operator -(BFloat16 left, BFloat16 right) =>
            new BFloat16((float)left - (float)right);

        public static BFloat16 operator *(BFloat16 left, BFloat16 right) =>
            new BFloat16((float)left * (float)right);

        public static BFloat16 operator /(BFloat16 left, BFloat16 right) =>
            new BFloat16((float)left / (float)right);

        // Comparison operators
        public static bool operator ==(BFloat16 left, BFloat16 right) =>
            left._value == right._value;

        public static bool operator !=(BFloat16 left, BFloat16 right) =>
            left._value != right._value;

        public static bool operator <(BFloat16 left, BFloat16 right) =>
            (float)left < (float)right;

        public static bool operator >(BFloat16 left, BFloat16 right) =>
            (float)left > (float)right;

        public static bool operator <=(BFloat16 left, BFloat16 right) =>
            (float)left <= (float)right;

        public static bool operator >=(BFloat16 left, BFloat16 right) =>
            (float)left >= (float)right;

        // Interface implementations
        public int CompareTo(BFloat16 other) =>
            ((float)this).CompareTo((float)other);

        public bool Equals(BFloat16 other) => _value == other._value;

        public override bool Equals(object obj) =>
            obj is BFloat16 other && Equals(other);

        public override int GetHashCode() => _value.GetHashCode();

        // Properties
        public bool IsNaN =>
            (_value & 0x7FFF) > 0x7F80 && (_value & 0x7FFF) < 0x8000;

        public bool IsInfinity =>
            (_value & 0x7FFF) == 0x7F80;

        public bool IsPositiveInfinity =>
            _value == PositiveInfinity._value;

        public bool IsNegativeInfinity =>
            _value == NegativeInfinity._value;

        // Constants
        public static readonly BFloat16 Epsilon = new BFloat16(0x007F);
        public static readonly BFloat16 MaxValue = new BFloat16(0x7F7F);
        public static readonly BFloat16 MinValue = new BFloat16(0xFF7F);
        public static readonly BFloat16 PositiveInfinity = new BFloat16(0x7F80);
        public static readonly BFloat16 NegativeInfinity = new BFloat16(0xFF80);
        public static readonly BFloat16 NaN = new BFloat16(0x7FC0);

        // Internal conversion methods
        private static ushort ConvertToBFloat16(float value)
        {
            unsafe
            {
                uint bits = *(uint*)&value;
                // Simply truncate the lower 16 bits (mantissa)
                // BFloat16 keeps the same exponent bits as FP32
                return (ushort)(bits >> 16);
            }
        }

        private static float ConvertToFloat(ushort bfloat)
        {
            unsafe
            {
                // Extend back to 32-bit by adding zeros to the mantissa
                uint bits = ((uint)bfloat) << 16;
                return *(float*)&bits;
            }
        }

        public override string ToString() => ((float)this).ToString();
    }
}
