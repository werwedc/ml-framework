namespace MLFramework.IR.LLIR.Operations.Memory
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.LLIR.Values;

    /// <summary>
    /// Memory copy operation in the Low-Level IR (LLIR).
    /// Copies data from source buffer to destination buffer.
    /// </summary>
    public class MemcpyOp : IROperation
    {
        /// <summary>Gets the destination buffer.</summary>
        public LLIRValue Dest { get; }

        /// <summary>Gets the source buffer.</summary>
        public LLIRValue Src { get; }

        /// <summary>Gets the size in bytes to copy.</summary>
        public int SizeInBytes { get; }

        /// <summary>
        /// Initializes a new instance of the MemcpyOp class.
        /// </summary>
        /// <param name="dest">The destination buffer.</param>
        /// <param name="src">The source buffer.</param>
        /// <param name="sizeInBytes">The size in bytes to copy.</param>
        public MemcpyOp(LLIRValue dest, LLIRValue src, int sizeInBytes)
            : base("memcpy", IROpcode.Memcpy, System.Array.Empty<MLFramework.IR.Values.IRValue>(), System.Array.Empty<MLFramework.IR.Types.IIRType>(), null)
        {
            Dest = dest ?? throw new System.ArgumentNullException(nameof(dest));
            Src = src ?? throw new System.ArgumentNullException(nameof(src));

            if (sizeInBytes <= 0)
            {
                throw new System.ArgumentOutOfRangeException(nameof(sizeInBytes), "Size must be positive.");
            }

            SizeInBytes = sizeInBytes;
        }

        public override void Validate()
        {
            // Validate that both dest and src are memory locations
            if (!Dest.IsMemoryLocation)
            {
                throw new System.InvalidOperationException("MemcpyOp requires destination to be a memory location, not a register.");
            }

            if (!Src.IsMemoryLocation)
            {
                throw new System.InvalidOperationException("MemcpyOp requires source to be a memory location, not a register.");
            }
        }

        public override IROperation Clone()
        {
            return new MemcpyOp(Dest, Src, SizeInBytes);
        }
    }
}
