namespace MLFramework.IR.Operations
{
    /// <summary>
    /// Represents the opcode for IR operations.
    /// </summary>
    public enum IROpcode
    {
        // High-level tensor operations
        /// <summary>Tensor addition</summary>
        Add,
        /// <summary>Tensor subtraction</summary>
        Sub,
        /// <summary>Tensor multiplication (element-wise)</summary>
        Mul,
        /// <summary>Tensor division</summary>
        Div,
        /// <summary>Matrix multiplication</summary>
        MatMul,
        /// <summary>2D convolution</summary>
        Conv2D,
        /// <summary>2D max pooling</summary>
        MaxPool2D,
        /// <summary>2D average pooling</summary>
        AvgPool2D,
        /// <summary>ReLU activation</summary>
        ReLU,
        /// <summary>Sigmoid activation</summary>
        Sigmoid,
        /// <summary>Tanh activation</summary>
        Tanh,
        /// <summary>GELU activation</summary>
        GELU,

        // Reduction operations
        /// <summary>Sum reduction</summary>
        ReduceSum,
        /// <summary>Mean reduction</summary>
        ReduceMean,
        /// <summary>Max reduction</summary>
        ReduceMax,
        /// <summary>Min reduction</summary>
        ReduceMin,
        /// <summary>Product reduction</summary>
        ReduceProd,

        // Shape operations
        /// <summary>Tensor reshape</summary>
        Reshape,
        /// <summary>Tensor transpose</summary>
        Transpose,
        /// <summary>Broadcast tensor to shape</summary>
        BroadcastTo,
        /// <summary>Tensor slicing</summary>
        Slice,
        /// <summary>Gather elements</summary>
        Gather,
        /// <summary>Scatter updates</summary>
        Scatter,
        /// <summary>Dynamic reshape</summary>
        DynamicReshape,

        // Control flow operations
        /// <summary>Conditional operation</summary>
        IfOp,
        /// <summary>Loop operation</summary>
        LoopOp,
        /// <summary>Scan/reduce operation</summary>
        ScanOp,
        /// <summary>For loop operation</summary>
        ForLoopOp,
        /// <summary>Parallel for loop operation</summary>
        ParallelForLoopOp,

        // Memory operations (LLIR)
        /// <summary>Allocate buffer</summary>
        AllocBuffer,
        /// <summary>Free buffer</summary>
        FreeBuffer,
        /// <summary>Load from memory</summary>
        Load,
        /// <summary>Store to memory</summary>
        Store,
        /// <summary>Memory copy</summary>
        Memcpy,
        /// <summary>Allocate tensor</summary>
        AllocTensor,
        /// <summary>Deallocate tensor</summary>
        DeallocTensor,

        // Arithmetic operations (scalar/vector)
        /// <summary>Scalar addition</summary>
        AddScalar,
        /// <summary>Scalar subtraction</summary>
        SubScalar,
        /// <summary>Scalar multiplication</summary>
        MulScalar,
        /// <summary>Scalar division</summary>
        DivScalar,
        /// <summary>Vector addition</summary>
        VectorAdd,
        /// <summary>Vector multiplication</summary>
        VectorMul,

        // Control flow operations (LLIR)
        /// <summary>Unconditional branch</summary>
        Branch,
        /// <summary>Conditional branch</summary>
        ConditionalBranch,
        /// <summary>Return from function</summary>
        Return,

        // Other operations
        /// <summary>Constant value</summary>
        Constant,
        /// <summary>Cast type</summary>
        Cast,
        /// <summary>Type conversion</summary>
        TypeConversion,

        // Placeholder for backend-specific opcodes
        /// <summary>Backend-specific operation</summary>
        BackendOp
    }
}
