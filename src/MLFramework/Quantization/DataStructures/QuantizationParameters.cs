namespace MLFramework.Quantization.DataStructures
{
    /// <summary>
    /// Quantization parameters for converting between floating-point and integer representations.
    /// </summary>
    public struct QuantizationParameters : IEquatable<QuantizationParameters>
    {
        /// <summary>
        /// Gets or sets the scale factor for quantization.
        /// </summary>
        public float Scale { get; set; }

        /// <summary>
        /// Gets or sets the zero-point offset.
        /// </summary>
        public int ZeroPoint { get; set; }

        /// <summary>
        /// Gets or sets the original minimum value (for calibration).
        /// </summary>
        public float Min { get; set; }

        /// <summary>
        /// Gets or sets the original maximum value (for calibration).
        /// </summary>
        public float Max { get; set; }

        /// <summary>
        /// Gets or sets the quantization mode.
        /// </summary>
        public QuantizationMode Mode { get; set; }

        /// <summary>
        /// Gets or sets the quantization type.
        /// </summary>
        public QuantizationType Type { get; set; }

        /// <summary>
        /// Gets or sets the channel-wise scale for per-channel quantization.
        /// Null for per-tensor quantization.
        /// </summary>
        public float[]? ChannelScales { get; set; }

        /// <summary>
        /// Gets or sets the channel-wise zero-point for per-channel quantization.
        /// Null for per-tensor quantization.
        /// </summary>
        public int[]? ChannelZeroPoints { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="QuantizationParameters"/> struct.
        /// </summary>
        /// <param name="scale">The scale factor.</param>
        /// <param name="zeroPoint">The zero-point offset.</param>
        /// <param name="min">The minimum value.</param>
        /// <param name="max">The maximum value.</param>
        /// <param name="mode">The quantization mode.</param>
        /// <param name="type">The quantization type.</param>
        public QuantizationParameters(
            float scale,
            int zeroPoint,
            float min = float.MinValue,
            float max = float.MaxValue,
            QuantizationMode mode = QuantizationMode.PerTensorSymmetric,
            QuantizationType type = QuantizationType.Int8)
        {
            Scale = scale;
            ZeroPoint = zeroPoint;
            Min = min;
            Max = max;
            Mode = mode;
            Type = type;
            ChannelScales = null;
            ChannelZeroPoints = null;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="QuantizationParameters"/> struct for per-channel quantization.
        /// </summary>
        /// <param name="channelScales">The channel-wise scales.</param>
        /// <param name="channelZeroPoints">The channel-wise zero-points.</param>
        /// <param name="min">The minimum value.</param>
        /// <param name="max">The maximum value.</param>
        /// <param name="mode">The quantization mode.</param>
        /// <param name="type">The quantization type.</param>
        public QuantizationParameters(
            float[] channelScales,
            int[] channelZeroPoints,
            float min = float.MinValue,
            float max = float.MaxValue,
            QuantizationMode mode = QuantizationMode.PerChannelSymmetric,
            QuantizationType type = QuantizationType.Int8)
        {
            ChannelScales = channelScales ?? throw new ArgumentNullException(nameof(channelScales));
            ChannelZeroPoints = channelZeroPoints ?? throw new ArgumentNullException(nameof(channelZeroPoints));

            if (channelScales.Length != channelZeroPoints.Length)
            {
                throw new ArgumentException("Channel scales and zero-points must have the same length.");
            }

            Scale = channelScales[0];
            ZeroPoint = channelZeroPoints[0];
            Min = min;
            Max = max;
            Mode = mode;
            Type = type;
        }

        /// <summary>
        /// Gets a value indicating whether this is per-channel quantization.
        /// </summary>
        public bool IsPerChannel => ChannelScales != null && ChannelZeroPoints != null;

        /// <summary>
        /// Gets the number of channels for per-channel quantization.
        /// Returns 1 for per-tensor quantization.
        /// </summary>
        public int ChannelCount => IsPerChannel ? ChannelScales!.Length : 1;

        /// <summary>
        /// Validates the quantization parameters.
        /// </summary>
        /// <returns>True if valid, false otherwise.</returns>
        public bool Validate()
        {
            if (Scale <= 0)
            {
                return false;
            }

            if (IsPerChannel)
            {
                if (ChannelScales!.Any(s => s <= 0))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Returns a string representation of the quantization parameters.
        /// </summary>
        public override string ToString()
        {
            if (IsPerChannel)
            {
                return $"QuantizationParameters({Mode}, {Type}, Channels: {ChannelCount}, " +
                       $"Scale range: [{ChannelScales!.Min():g4}, {ChannelScales!.Max():g4}], " +
                       $"ZeroPoint range: [{ChannelZeroPoints!.Min()}, {ChannelZeroPoints!.Max()}])";
            }

            return $"QuantizationParameters({Mode}, {Type}, Scale: {Scale:g4}, ZeroPoint: {ZeroPoint}, " +
                   $"Range: [{Min:g4}, {Max:g4}])";
        }

        /// <summary>
        /// Determines whether the specified parameters are equal to the current instance.
        /// </summary>
        public bool Equals(QuantizationParameters other)
        {
            if (Scale != other.Scale || ZeroPoint != other.ZeroPoint ||
                Min != other.Min || Max != other.Max ||
                Mode != other.Mode || Type != other.Type)
            {
                return false;
            }

            if (IsPerChannel && other.IsPerChannel)
            {
                if (ChannelScales!.Length != other.ChannelScales!.Length)
                {
                    return false;
                }

                for (int i = 0; i < ChannelScales!.Length; i++)
                {
                    if (ChannelScales[i] != other.ChannelScales[i] ||
                        ChannelZeroPoints![i] != other.ChannelZeroPoints![i])
                    {
                        return false;
                    }
                }
            }

            return !IsPerChannel && !other.IsPerChannel;
        }

        /// <summary>
        /// Determines whether the specified object is equal to the current instance.
        /// </summary>
        public override bool Equals(object? obj)
        {
            return obj is QuantizationParameters other && Equals(other);
        }

        /// <summary>
        /// Returns a hash code for the current instance.
        /// </summary>
        public override int GetHashCode()
        {
            var hash = HashCode.Combine(Scale, ZeroPoint, Min, Max, Mode, Type);

            if (IsPerChannel)
            {
                foreach (var scale in ChannelScales!)
                {
                    hash = HashCode.Combine(hash, scale);
                }

                foreach (var zp in ChannelZeroPoints!)
                {
                    hash = HashCode.Combine(hash, zp);
                }
            }

            return hash;
        }

        /// <summary>
        /// Determines whether two parameters are equal.
        /// </summary>
        public static bool operator ==(QuantizationParameters left, QuantizationParameters right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Determines whether two parameters are not equal.
        /// </summary>
        public static bool operator !=(QuantizationParameters left, QuantizationParameters right)
        {
            return !(left == right);
        }
    }
}
