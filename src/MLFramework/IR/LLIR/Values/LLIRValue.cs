namespace MLFramework.IR.LLIR.Values
{
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;

    /// <summary>
    /// Represents a value in the Low-Level IR (LLIR).
    /// LLIR values can be either registers or memory locations.
    /// </summary>
    public class LLIRValue : IRValue
    {
        /// <summary>Gets whether this value is a register.</summary>
        public bool IsRegister { get; }

        /// <summary>Gets whether this value is a memory location.</summary>
        public bool IsMemoryLocation { get; }

        /// <summary>
        /// Initializes a new instance of the LLIRValue class.
        /// </summary>
        /// <param name="type">The type of this value.</param>
        /// <param name="name">The name of this value.</param>
        /// <param name="isRegister">Whether this value is a register (true) or memory location (false).</param>
        public LLIRValue(IIRType type, string name, bool isRegister = false)
            : base(type, name)
        {
            IsRegister = isRegister;
            IsMemoryLocation = !isRegister;
        }

        public override string ToString()
        {
            return $"{Name} : {Type} ({(IsRegister ? "reg" : "mem")})";
        }
    }
}
