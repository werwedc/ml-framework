using System;

namespace MLFramework.IR.MLIR.Shape
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Values;
    using MLFramework.IR.Types;

    /// <summary>
    /// Explicit transpose operation in the Mid-Level IR (MLIR).
    /// </summary>
    public class TransposeOp : IROperation
    {
        /// <summary>Gets the input tensor.</summary>
        public IRValue Input { get; }

        /// <summary>Gets the result tensor.</summary>
        public IRValue Result => Results[0];

        /// <summary>Gets the permutation of dimensions.</summary>
        public int[] Permutation { get; }

        /// <summary>Gets the input shape (explicit).</summary>
        public int[] InputShape { get; }

        /// <summary>Gets the output shape (explicit).</summary>
        public int[] OutputShape { get; }

        /// <summary>
        /// Initializes a new instance of the TransposeOp class.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="result">The result tensor.</param>
        /// <param name="permutation">The permutation of dimensions.</param>
        /// <param name="inputShape">The explicit input shape.</param>
        /// <param name="outputShape">The explicit output shape.</param>
        public TransposeOp(IRValue input, IRValue result, int[] permutation,
                         int[] inputShape, int[] outputShape)
            : base("transpose", IROpcode.Transpose, new[] { input }, new[] { result.Type }, null)
        {
            Input = input ?? throw new ArgumentNullException(nameof(input));
            Permutation = permutation ?? throw new ArgumentNullException(nameof(permutation));
            InputShape = inputShape ?? throw new ArgumentNullException(nameof(inputShape));
            OutputShape = outputShape ?? throw new ArgumentNullException(nameof(outputShape));
            Results[0] = result ?? throw new ArgumentNullException(nameof(result));
        }

        public override void Validate()
        {
            if (Input.Type is not TensorType inputType)
                throw new InvalidOperationException("Input must be a tensor type");

            if (Permutation.Length != InputShape.Length)
                throw new InvalidOperationException("Permutation length must match input rank");
        }

        public override IROperation Clone()
        {
            return new TransposeOp(Input, Result, Permutation, InputShape, OutputShape);
        }
    }
}
