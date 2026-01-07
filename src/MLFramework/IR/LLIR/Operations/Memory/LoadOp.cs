namespace MLFramework.IR.LLIR.Operations.Memory
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.LLIR.Values;

    /// <summary>
    /// Load operation in the Low-Level IR (LLIR).
    /// Loads a value from a memory address into a register.
    /// </summary>
    public class LoadOp : IROperation
    {
        /// <summary>Gets the memory address to load from.</summary>
        public LLIRValue Address { get; }

        /// <summary>Gets the result register containing the loaded value.</summary>
        public LLIRValue Result => Results[0] as LLIRValue;

        /// <summary>Gets the offset in bytes from the base address.</summary>
        public int Offset { get; }

        /// <summary>
        /// Initializes a new instance of the LoadOp class.
        /// </summary>
        /// <param name="address">The memory address to load from.</param>
        /// <param name="result">The result register.</param>
        /// <param name="offset">The offset in bytes from the base address.</param>
        public LoadOp(LLIRValue address, LLIRValue result, int offset = 0)
            : base("load", IROpcode.Load, new[] { address }, new[] { result.Type }, null)
        {
            Address = address ?? throw new System.ArgumentNullException(nameof(address));
            Results[0] = result ?? throw new System.ArgumentNullException(nameof(result));
            Offset = offset;
        }

        public override void Validate()
        {
            // Validate address and result types
        }

        public override IROperation Clone()
        {
            return new LoadOp(Address, Result, Offset);
        }
    }
}
