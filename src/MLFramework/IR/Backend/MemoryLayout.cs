namespace MLFramework.IR.Backend
{
    /// <summary>
    /// Memory layout options for tensor data
    /// </summary>
    public enum MemoryLayout
    {
        /// <summary>
        /// Row-major order (last dimension varies fastest)
        /// </summary>
        RowMajor,

        /// <summary>
        /// Column-major order (first dimension varies fastest)
        /// </summary>
        ColumnMajor
    }
}
