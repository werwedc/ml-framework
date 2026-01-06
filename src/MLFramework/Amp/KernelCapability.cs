namespace MLFramework.Amp
{
    /// <summary>
    /// Describes the capabilities of a GPU kernel
    /// </summary>
    public class KernelCapability
    {
        /// <summary>
        /// Gets the kernel data type
        /// </summary>
        public KernelDtype Dtype { get; }

        /// <summary>
        /// Gets whether the kernel is available on the device
        /// </summary>
        public bool IsAvailable { get; }

        /// <summary>
        /// Gets whether the kernel supports tensor cores (if applicable)
        /// </summary>
        public bool SupportsTensorCores { get; }

        /// <summary>
        /// Gets the relative performance factor (higher = faster)
        /// </summary>
        public float PerformanceFactor { get; }

        /// <summary>
        /// Gets the memory efficiency factor (higher = more memory efficient)
        /// </summary>
        public float MemoryEfficiency { get; }

        /// <summary>
        /// Creates a new KernelCapability
        /// </summary>
        public KernelCapability(
            KernelDtype dtype,
            bool isAvailable = true,
            bool supportsTensorCores = false,
            float performanceFactor = 1.0f,
            float memoryEfficiency = 1.0f)
        {
            Dtype = dtype;
            IsAvailable = isAvailable;
            SupportsTensorCores = supportsTensorCores;
            PerformanceFactor = performanceFactor;
            MemoryEfficiency = memoryEfficiency;
        }

        /// <summary>
        /// Creates a kernel capability for FP32
        /// </summary>
        public static KernelCapability CreateFloat32(bool supportsTensorCores = false)
        {
            return new KernelCapability(
                KernelDtype.Float32,
                isAvailable: true,
                supportsTensorCores: supportsTensorCores,
                performanceFactor: 1.0f,
                memoryEfficiency: 1.0f);
        }

        /// <summary>
        /// Creates a kernel capability for FP16
        /// </summary>
        public static KernelCapability CreateFloat16(bool supportsTensorCores = true)
        {
            return new KernelCapability(
                KernelDtype.Float16,
                isAvailable: true,
                supportsTensorCores: supportsTensorCores,
                performanceFactor: 2.0f,
                memoryEfficiency: 2.0f);
        }

        /// <summary>
        /// Creates a kernel capability for BF16
        /// </summary>
        public static KernelCapability CreateBFloat16(bool supportsTensorCores = true)
        {
            return new KernelCapability(
                KernelDtype.BFloat16,
                isAvailable: true,
                supportsTensorCores: supportsTensorCores,
                performanceFactor: 2.0f,
                memoryEfficiency: 2.0f);
        }

        public override string ToString()
        {
            return $"KernelCapability(" +
                   $"Dtype={Dtype}, " +
                   $"Available={IsAvailable}, " +
                   $"TensorCores={SupportsTensorCores}, " +
                   $"PerfFactor={PerformanceFactor:F2}, " +
                   $"MemEfficiency={MemoryEfficiency:F2})";
        }
    }
}
