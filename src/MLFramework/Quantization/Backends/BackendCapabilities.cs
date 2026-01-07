namespace MLFramework.Quantization.Backends
{
    /// <summary>
    /// Backend capabilities descriptor.
    /// </summary>
    public struct BackendCapabilities : IEquatable<BackendCapabilities>
    {
        /// <summary>
        /// Gets or sets the capability flags.
        /// </summary>
        public BackendCapabilityFlags Flags { get; set; }

        /// <summary>
        /// Gets or sets the maximum tensor size supported (in bytes).
        /// </summary>
        public long MaxTensorSize { get; set; }

        /// <summary>
        /// Gets or sets the minimum batch size for optimal performance.
        /// </summary>
        public int MinBatchSize { get; set; }

        /// <summary>
        /// Gets or sets the preferred batch size for optimal performance.
        /// </summary>
        public int PreferredBatchSize { get; set; }

        /// <summary>
        /// Gets or sets the maximum number of threads/workers supported.
        /// </summary>
        public int MaxThreads { get; set; }

        /// <summary>
        /// Gets a value indicating whether Int8 matrix multiplication is supported.
        /// </summary>
        public bool SupportsInt8MatMul => (Flags & BackendCapabilityFlags.Int8MatMul) != 0;

        /// <summary>
        /// Gets a value indicating whether Int8 2D convolution is supported.
        /// </summary>
        public bool SupportsInt8Conv2D => (Flags & BackendCapabilityFlags.Int8Conv2D) != 0;

        /// <summary>
        /// Gets a value indicating whether per-channel quantization is supported.
        /// </summary>
        public bool SupportsPerChannelQuantization => (Flags & BackendCapabilityFlags.PerChannelQuantization) != 0;

        /// <summary>
        /// Gets a value indicating whether mixed precision is supported.
        /// </summary>
        public bool SupportsMixedPrecision => (Flags & BackendCapabilityFlags.MixedPrecision) != 0;

        /// <summary>
        /// Gets a value indicating whether dynamic quantization is supported.
        /// </summary>
        public bool SupportsDynamicQuantization => (Flags & BackendCapabilityFlags.DynamicQuantization) != 0;

        /// <summary>
        /// Gets a value indicating whether static quantization is supported.
        /// </summary>
        public bool SupportsStaticQuantization => (Flags & BackendCapabilityFlags.StaticQuantization) != 0;

        /// <summary>
        /// Gets a value indicating whether asymmetric quantization is supported.
        /// </summary>
        public bool SupportsAsymmetricQuantization => (Flags & BackendCapabilityFlags.AsymmetricQuantization) != 0;

        /// <summary>
        /// Gets a value indicating whether symmetric quantization is supported.
        /// </summary>
        public bool SupportsSymmetricQuantization => (Flags & BackendCapabilityFlags.SymmetricQuantization) != 0;

        /// <summary>
        /// Initializes a new instance of the <see cref="BackendCapabilities"/> struct.
        /// </summary>
        public BackendCapabilities(
            BackendCapabilityFlags flags,
            long maxTensorSize = long.MaxValue,
            int minBatchSize = 1,
            int preferredBatchSize = 32,
            int maxThreads = 0)
        {
            Flags = flags;
            MaxTensorSize = maxTensorSize;
            MinBatchSize = minBatchSize;
            PreferredBatchSize = preferredBatchSize;
            MaxThreads = maxThreads == 0 ? Environment.ProcessorCount : maxThreads;
        }

        /// <summary>
        /// Returns a string representation of the capabilities.
        /// </summary>
        public override string ToString()
        {
            var capabilities = new List<string>();

            if (SupportsInt8MatMul) capabilities.Add("Int8MatMul");
            if (SupportsInt8Conv2D) capabilities.Add("Int8Conv2D");
            if (SupportsPerChannelQuantization) capabilities.Add("PerChannelQuantization");
            if (SupportsMixedPrecision) capabilities.Add("MixedPrecision");
            if (SupportsDynamicQuantization) capabilities.Add("DynamicQuantization");
            if (SupportsStaticQuantization) capabilities.Add("StaticQuantization");
            if (SupportsAsymmetricQuantization) capabilities.Add("AsymmetricQuantization");
            if (SupportsSymmetricQuantization) capabilities.Add("SymmetricQuantization");

            return $"Capabilities: [{string.Join(", ", capabilities)}], " +
                   $"MaxTensorSize: {MaxTensorSize}, " +
                   $"MinBatchSize: {MinBatchSize}, " +
                   $"PreferredBatchSize: {PreferredBatchSize}, " +
                   $"MaxThreads: {MaxThreads}";
        }

        /// <summary>
        /// Determines whether the specified capabilities are equal to the current instance.
        /// </summary>
        public bool Equals(BackendCapabilities other)
        {
            return Flags == other.Flags &&
                   MaxTensorSize == other.MaxTensorSize &&
                   MinBatchSize == other.MinBatchSize &&
                   PreferredBatchSize == other.PreferredBatchSize &&
                   MaxThreads == other.MaxThreads;
        }

        /// <summary>
        /// Determines whether the specified object is equal to the current instance.
        /// </summary>
        public override bool Equals(object? obj)
        {
            return obj is BackendCapabilities other && Equals(other);
        }

        /// <summary>
        /// Returns a hash code for the current instance.
        /// </summary>
        public override int GetHashCode()
        {
            return HashCode.Combine(
                Flags,
                MaxTensorSize,
                MinBatchSize,
                PreferredBatchSize,
                MaxThreads);
        }

        /// <summary>
        /// Determines whether two capabilities are equal.
        /// </summary>
        public static bool operator ==(BackendCapabilities left, BackendCapabilities right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Determines whether two capabilities are not equal.
        /// </summary>
        public static bool operator !=(BackendCapabilities left, BackendCapabilities right)
        {
            return !(left == right);
        }
    }
}
