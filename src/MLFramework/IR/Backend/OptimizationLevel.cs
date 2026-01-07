namespace MLFramework.IR.Backend
{
    /// <summary>
    /// Optimization levels for compilation
    /// </summary>
    public enum OptimizationLevel
    {
        /// <summary>
        /// No optimizations
        /// </summary>
        None,

        /// <summary>
        /// Basic optimizations only
        /// </summary>
        Basic,

        /// <summary>
        /// Standard optimization level (default)
        /// </summary>
        Standard,

        /// <summary>
        /// Aggressive optimizations
        /// </summary>
        Aggressive
    }
}
