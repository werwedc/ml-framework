namespace MLFramework.IR.MLIR.Reduce
{
    /// <summary>
    /// Represents the kind of reduction operation.
    /// </summary>
    public enum ReductionKind
    {
        /// <summary>Sum reduction</summary>
        Sum,
        /// <summary>Mean reduction</summary>
        Mean,
        /// <summary>Max reduction</summary>
        Max,
        /// <summary>Min reduction</summary>
        Min,
        /// <summary>Product reduction</summary>
        Prod,
        /// <summary>Any reduction (logical OR)</summary>
        Any,
        /// <summary>All reduction (logical AND)</summary>
        All
    }
}
