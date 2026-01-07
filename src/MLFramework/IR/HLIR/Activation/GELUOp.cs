using System;

namespace MLFramework.IR.HLIR.Activation
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;

    /// <summary>
    /// Gaussian Error Linear Unit (GELU) activation operation.
    /// </summary>
    public class GELUOp : IROperation
    {
        /// <summary>Gets the input tensor.</summary>
        public IRValue Input { get; }

        /// <summary>Gets the output tensor.</summary>
        public IRValue Result => Results[0];

        /// <summary>
        /// Initializes a new instance of the GELUOp class.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="result">The output tensor.</param>
        public GELUOp(IRValue input, IRValue result)
            : base("gelu", IROpcode.GELU, new[] { input }, new[] { result.Type }, null)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            Input = input;
            Results[0] = result;
        }

        /// <summary>
        /// Validates the operation.
        /// </summary>
        public override void Validate()
        {
            if (Input.Type is not TensorType inputType)
                throw new InvalidOperationException("Input must be a tensor type");
            if (Result.Type is not TensorType resultType)
                throw new InvalidOperationException("Result must be a tensor type");

            // GELU preserves shape and type
            if (inputType.ElementType != resultType.ElementType)
                throw new InvalidOperationException($"Element types must match: {inputType.ElementType} vs {resultType.ElementType}");

            if (inputType.Shape.Length != resultType.Shape.Length)
                throw new InvalidOperationException("Input and result must have same rank");

            for (int i = 0; i < inputType.Shape.Length; i++)
            {
                if (inputType.Shape[i] != resultType.Shape[i])
                    throw new InvalidOperationException($"Input and result shapes must match at dimension {i}");
            }
        }

        /// <summary>
        /// Creates a new GELUOp with auto-generated result.
        /// </summary>
        public static IRValue Create(IRContext ctx, IRValue input, string name = null)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            var inputType = (TensorType)input.Type;
            var resultType = new TensorType(inputType.ElementType, (int[])inputType.Shape.Clone());
            var result = ctx.CreateValue(resultType, name);

            var op = new GELUOp(input, result);
            ctx.RegisterOperation(op);
            return result;
        }

        /// <summary>
        /// Creates a clone of this operation.
        /// </summary>
        public override IROperation Clone()
        {
            return new GELUOp(Input, Result);
        }
    }
}
