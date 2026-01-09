namespace MLFramework.MobileRuntime.Backends.Cpu.Models
{
    /// <summary>
    /// Parameters for 2D pooling operations.
    /// </summary>
    public class Pool2DParams
    {
        /// <summary>
        /// Kernel size [height, width].
        /// </summary>
        public int[] KernelSize { get; set; } = new int[2];

        /// <summary>
        /// Stride [height, width].
        /// </summary>
        public int[] Stride { get; set; } = new int[2];

        /// <summary>
        /// Padding [height, width].
        /// </summary>
        public int[] Padding { get; set; } = new int[2];

        /// <summary>
        /// Whether to include padding in the count for average pooling.
        /// </summary>
        public bool CountIncludePad { get; set; } = true;
    }
}
