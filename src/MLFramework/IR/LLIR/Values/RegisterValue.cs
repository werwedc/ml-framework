namespace MLFramework.IR.LLIR.Values
{
    using MLFramework.IR.Types;

    /// <summary>
    /// Represents a register value in the Low-Level IR (LLIR).
    /// Register values are fast, temporary storage locations used during computation.
    /// </summary>
    public class RegisterValue : LLIRValue
    {
        /// <summary>
        /// Initializes a new instance of the RegisterValue class.
        /// </summary>
        /// <param name="type">The type of this register.</param>
        /// <param name="name">The name of this register.</param>
        public RegisterValue(IIRType type, string name)
            : base(type, name, isRegister: true)
        {
        }
    }
}
