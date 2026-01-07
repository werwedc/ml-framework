namespace MLFramework.IR.Backend
{
    /// <summary>
    /// Options for compilation
    /// </summary>
    public class CompilationOptions
    {
        /// <summary>
        /// Optimization level to apply
        /// </summary>
        public OptimizationLevel OptimizationLevel { get; set; } = OptimizationLevel.Standard;

        /// <summary>
        /// Generate debug symbols
        /// </summary>
        public bool DebugSymbols { get; set; } = false;

        /// <summary>
        /// Verbose output
        /// </summary>
        public bool Verbose { get; set; } = false;

        /// <summary>
        /// Vector width (-1 means auto-detect)
        /// </summary>
        public int VectorWidth { get; set; } = -1;

        /// <summary>
        /// Preferred memory layout
        /// </summary>
        public MemoryLayout PreferredMemoryLayout { get; set; } = MemoryLayout.RowMajor;
    }
}
