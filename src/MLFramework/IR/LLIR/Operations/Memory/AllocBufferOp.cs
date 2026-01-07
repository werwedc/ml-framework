namespace MLFramework.IR.LLIR.Operations.Memory
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.LLIR.Values;

    /// <summary>
    /// Buffer allocation operation in the Low-Level IR (LLIR).
    /// Allocates a buffer in memory for storing data.
    /// </summary>
    public class AllocBufferOp : IROperation
    {
        /// <summary>Gets the buffer value representing the allocated memory.</summary>
        public LLIRValue Buffer { get; }

        /// <summary>Gets the size of the buffer in bytes.</summary>
        public int SizeInBytes { get; }

        /// <summary>Gets the alignment requirement in bytes.</summary>
        public int Alignment { get; }

        /// <summary>
        /// Initializes a new instance of the AllocBufferOp class.
        /// </summary>
        /// <param name="buffer">The buffer value representing the allocated memory.</param>
        /// <param name="sizeInBytes">The size of the buffer in bytes.</param>
        /// <param name="alignment">The alignment requirement in bytes.</param>
        public AllocBufferOp(LLIRValue buffer, int sizeInBytes, int alignment = 16)
            : base("alloc_buffer", IROpcode.AllocBuffer, System.Array.Empty<MLFramework.IR.Values.IRValue>(), new[] { buffer.Type }, null)
        {
            Buffer = buffer ?? throw new System.ArgumentNullException(nameof(buffer));
            SizeInBytes = sizeInBytes;
            Alignment = alignment > 0 ? alignment : 16;
            Results[0] = buffer;
        }

        public override void Validate()
        {
            // Validate buffer type and alignment
        }

        public override IROperation Clone()
        {
            return new AllocBufferOp(Buffer, SizeInBytes, Alignment);
        }
    }
}
