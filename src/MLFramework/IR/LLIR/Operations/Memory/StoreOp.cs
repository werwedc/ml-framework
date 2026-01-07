namespace MLFramework.IR.LLIR.Operations.Memory
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.LLIR.Values;

    /// <summary>
    /// Store operation in the Low-Level IR (LLIR).
    /// Stores a value from a register to a memory address.
    /// </summary>
    public class StoreOp : IROperation
    {
        /// <summary>Gets the memory address to store to.</summary>
        public LLIRValue Address { get; }

        /// <summary>Gets the value to store.</summary>
        public LLIRValue Value { get; }

        /// <summary>Gets the offset in bytes from the base address.</summary>
        public int Offset { get; }

        /// <summary>
        /// Initializes a new instance of the StoreOp class.
        /// </summary>
        /// <param name="address">The memory address to store to.</param>
        /// <param name="value">The value to store.</param>
        /// <param name="offset">The offset in bytes from the base address.</param>
        public StoreOp(LLIRValue address, LLIRValue value, int offset = 0)
            : base("store", IROpcode.Store, new[] { address, value }, System.Array.Empty<MLFramework.IR.Types.IIRType>(), null)
        {
            Address = address ?? throw new System.ArgumentNullException(nameof(address));
            Value = value ?? throw new System.ArgumentNullException(nameof(value));
            Offset = offset;
        }

        public override void Validate()
        {
            // Validate address and value types
        }

        public override IROperation Clone()
        {
            return new StoreOp(Address, Value, Offset);
        }
    }
}
