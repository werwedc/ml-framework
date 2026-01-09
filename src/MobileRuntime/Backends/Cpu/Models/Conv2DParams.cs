namespace MLFramework.MobileRuntime.Backends.Cpu.Models
{
    /// <summary>
    /// Parameters for 2D convolution operations.
    /// </summary>
    public class Conv2DParams
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
        /// Dilation [height, width].
        /// </summary>
        public int[] Dilation { get; set; } = new int[] { 1, 1 };

        /// <summary>
        /// Number of groups for grouped convolution.
        /// </summary>
        public int Groups { get; set; } = 1;
    }
}
