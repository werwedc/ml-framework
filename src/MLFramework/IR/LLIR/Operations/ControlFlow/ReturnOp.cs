namespace MLFramework.IR.LLIR.Operations.ControlFlow
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.LLIR.Values;

    /// <summary>
    /// Return operation in the Low-Level IR (LLIR).
    /// Returns from a function with an optional return value.
    /// </summary>
    public class ReturnOp : IROperation
    {
        /// <summary>Gets the return value (can be null for void functions).</summary>
        public LLIRValue ReturnValue { get; }

        /// <summary>
        /// Initializes a new instance of the ReturnOp class with a return value.
        /// </summary>
        /// <param name="returnValue">The return value.</param>
        public ReturnOp(LLIRValue returnValue)
            : this(returnValue, true)
        {
        }

        /// <summary>
        /// Initializes a new instance of the ReturnOp class (void return).
        /// </summary>
        public ReturnOp()
            : this(null, false)
        {
        }

        private ReturnOp(LLIRValue returnValue, bool hasReturnValue)
            : base("return", IROpcode.Return,
                  hasReturnValue ? new[] { returnValue } : System.Array.Empty<MLFramework.IR.Values.IRValue>(),
                  System.Array.Empty<MLFramework.IR.Types.IIRType>(), null)
        {
            ReturnValue = returnValue;
        }

        public override void Validate()
        {
            // Validate return type matches function signature (if known)
        }

        public override IROperation Clone()
        {
            return ReturnValue != null ? new ReturnOp(ReturnValue) : new ReturnOp();
        }
    }
}
