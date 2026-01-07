namespace MLFramework.IR.LLIR.Operations.Memory
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.LLIR.Values;

    /// <summary>
    /// Buffer deallocation operation in the Low-Level IR (LLIR).
    /// Frees a previously allocated buffer in memory.
    /// </summary>
    public class FreeBufferOp : IROperation
    {
        /// <summary>Gets the buffer value to free.</summary>
        public LLIRValue Buffer { get; }

        /// <summary>
        /// Initializes a new instance of the FreeBufferOp class.
        /// </summary>
        /// <param name="buffer">The buffer value to free.</param>
        public FreeBufferOp(LLIRValue buffer)
            : base("free_buffer", IROpcode.FreeBuffer, System.Array.Empty<MLFramework.IR.Values.IRValue>(), System.Array.Empty<MLFramework.IR.Types.IIRType>(), null)
        {
            Buffer = buffer ?? throw new System.ArgumentNullException(nameof(buffer));
        }

        public override void Validate()
        {
            // Validate that buffer is a memory location
            if (!Buffer.IsMemoryLocation)
            {
                throw new System.InvalidOperationException("FreeBufferOp requires a memory location, not a register.");
            }
        }

        public override IROperation Clone()
        {
            return new FreeBufferOp(Buffer);
        }
    }
}
