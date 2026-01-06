using System;

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
        public Half(float value) : this(ConvertToHalf(value)) { }

        public Half(double value) : this(ConvertToHalf((float)value)) { }

        public Half(ushort rawBits)
        {
            _value = rawBits;
        }

        // Conversion operators
        public static explicit operator float(Half h) => ConvertToFloat(h._value);

        public static explicit operator double(Half h) => ConvertToFloat(h._value);

        public static explicit operator Half(float f) => new Half(f);

        public static explicit operator Half(double d) => new Half((float)d);

        // Arithmetic operators
        public static Half operator +(Half left, Half right) =>
            new Half((float)left + (float)right);

        public static Half operator -(Half left, Half right) =>
            new Half((float)left - (float)right);

        public static Half operator *(Half left, Half right) =>
            new Half((float)left * (float)right);

        public static Half operator /(Half left, Half right) =>
            new Half((float)left / (float)right);

        // Comparison operators
        public static bool operator ==(Half left, Half right) => left._value == right._value;

        public static bool operator !=(Half left, Half right) => left._value != right._value;

        public static bool operator <(Half left, Half right) =>
            (float)left < (float)right;

        public static bool operator >(Half left, Half right) =>
            (float)left > (float)right;

        public static bool operator <=(Half left, Half right) =>
            (float)left <= (float)right;

        public static bool operator >=(Half left, Half right) =>
            (float)left >= (float)right;

        // Interface implementations
        public int CompareTo(Half other) => ((float)this).CompareTo((float)other);

        public bool Equals(Half other) => _value == other._value;

        public override bool Equals(object obj) =>
            obj is Half other && Equals(other);

        public override int GetHashCode() => _value.GetHashCode();

        // Properties
        public bool IsNaN =>
            (_value & 0x7FFF) > 0x7C00 && (_value & 0x7FFF) < 0x8000;

        public bool IsInfinity =>
            (_value & 0x7FFF) == 0x7C00;

        public bool IsPositiveInfinity =>
            _value == PositiveInfinity._value;

        public bool IsNegativeInfinity =>
            _value == NegativeInfinity._value;

        // Constants
        public static readonly Half Epsilon = new Half(0x0001);
        public static readonly Half MaxValue = new Half(0x7BFF);
        public static readonly Half MinValue = new Half(0xFBFF);
        public static readonly Half PositiveInfinity = new Half(0x7C00);
        public static readonly Half NegativeInfinity = new Half(0xFC00);
        public static readonly Half NaN = new Half(0x7E00);

        // Internal conversion methods
        private static ushort ConvertToHalf(float value)
        {
            unsafe
            {
                uint bits = *(uint*)&value;
                uint sign = (bits >> 16) & 0x8000;
                int exponent = (int)((bits >> 23) & 0xFF) - 127 + 15;
                uint mantissa = bits & 0x7FFFFF;

                if (exponent <= 0)
                {
                    if (exponent < -10)
                    {
                        // Too small, round to zero
                        return (ushort)sign;
                    }

                    // Denormal
                    mantissa |= 0x800000;
                    int shift = 14 - exponent;
                    mantissa >>= shift;
                    return (ushort)(sign | (mantissa >> 13));
                }

                if (exponent >= 31)
                {
                    if (exponent == 128 && mantissa != 0)
                    {
                        // NaN
                        return (ushort)(sign | 0x7E00);
                    }
                    // Infinity
                    return (ushort)(sign | 0x7C00);
                }

                mantissa >>= 13;
                return (ushort)(sign | ((uint)exponent << 10) | mantissa);
            }
        }

        private static float ConvertToFloat(ushort half)
        {
            unsafe
            {
                uint sign = ((uint)half >> 15) << 31;
                uint exponent = (uint)((half >> 10) & 0x1F);
                uint mantissa = (uint)(half & 0x3FF);

                if (exponent == 0)
                {
                    if (mantissa == 0)
                    {
                        // Zero
                        return *(float*)&sign;
                    }

                    // Denormal
                    mantissa <<= 13;
                    while ((mantissa & 0x800000) == 0)
                    {
                        mantissa <<= 1;
                        exponent--;
                    }
                    exponent++;
                    mantissa &= 0x7FFFFF;
                }
                else if (exponent == 31)
                {
                    // Infinity or NaN
                    mantissa <<= 13;
                    uint resultInf = sign | 0x7F800000 | mantissa;
                    return *(float*)&resultInf;
                }

                exponent += 112;
                uint result = sign | ((uint)exponent << 23) | (mantissa << 13);
                return *(float*)&result;
            }
        }

        public override string ToString() => ((float)this).ToString();
    }
}
