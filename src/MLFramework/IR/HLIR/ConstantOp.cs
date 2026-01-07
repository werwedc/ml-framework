using System;

namespace MLFramework.IR.HLIR
{
    using MLFramework.IR.Attributes;
    using MLFramework.IR.Operations;
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;

    /// <summary>
    /// Constant operation that stores a constant value.
    /// </summary>
    public class ConstantOp : IROperation
    {
        /// <summary>Gets the constant value.</summary>
        public IIRAttribute Value { get; }

        /// <summary>Gets the output value.</summary>
        private IRValue _output;
        public IRValue Output => _output ?? Results[0];

        /// <summary>
        /// Initializes a new instance of the ConstantOp class.
        /// </summary>
        /// <param name="output">The output value.</param>
        /// <param name="value">The constant value.</param>
        public ConstantOp(IRValue output, IIRAttribute value)
            : base("constant", IROpcode.Constant, Array.Empty<IRValue>(), new[] { output.Type }, null)
        {
            if (output == null)
                throw new ArgumentNullException(nameof(output));
            if (value == null)
                throw new ArgumentNullException(nameof(value));

            _output = output;
            Value = value;
        }

        /// <summary>
        /// Validates the operation.
        /// </summary>
        public override void Validate()
        {
            if (Value.Type is not TensorType valueType && Value.Type is not ScalarType)
                throw new InvalidOperationException("Constant value must be a tensor or scalar type");

            if (!Output.Type.Equals(Value.Type))
                throw new InvalidOperationException($"Output type {Output.Type} does not match constant type {Value.Type}");
        }

        /// <summary>
        /// Creates a new ConstantOp with auto-generated output.
        /// </summary>
        public static IRValue Create(IRContext ctx, IIRAttribute value, string name = null)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));

            var result = ctx.CreateValue(value.Type, name);
            var op = new ConstantOp(result, value);
            ctx.RegisterOperation(op);
            return result;
        }

        /// <summary>
        /// Creates a clone of this operation.
        /// </summary>
        public override IROperation Clone()
        {
            return new ConstantOp(Output, Value);
        }
    }
}
