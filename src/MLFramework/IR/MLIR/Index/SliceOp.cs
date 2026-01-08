using System;
using MLFramework.IR.Operations;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.IR.MLIR.Index
{
    /// <summary>
    /// Slice operation in MLIR.
    /// Extracts a slice from a tensor using start, end, and stride indices.
    /// </summary>
    public class SliceOp : IROperation
    {
        /// <summary>Gets the input tensor.</summary>
        public IRValue Input => Operands[0];

        /// <summary>Gets the result tensor (slice).</summary>
        public IRValue Result => Results[0];

        /// <summary>Gets the start indices for each dimension.</summary>
        public int[] Starts { get; }

        /// <summary>Gets the end indices for each dimension.</summary>
        public int[] Ends { get; }

        /// <summary>Gets the stride for each dimension.</summary>
        public int[] Strides { get; }

        /// <summary>
        /// Initializes a new instance of the SliceOp class.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="result">The result value.</param>
        /// <param name="starts">The start indices for each dimension.</param>
        /// <param name="ends">The end indices for each dimension.</param>
        /// <param name="strides">The stride for each dimension.</param>
        public SliceOp(IRValue input, IRValue result,
                      int[] starts, int[] ends, int[] strides)
            : base("slice", IROpcode.Slice, new[] { input }, new[] { result.Type }, null)
        {
            Results[0] = result ?? throw new ArgumentNullException(nameof(result));
            Starts = starts ?? throw new ArgumentNullException(nameof(starts));
            Ends = ends ?? throw new ArgumentNullException(nameof(ends));
            Strides = strides ?? throw new ArgumentNullException(nameof(strides));
        }

        /// <summary>
        /// Validates the slice operation.
        /// </summary>
        public override void Validate()
        {
            // Validate that input is a tensor
            if (!(Input.Type is TensorType inputType))
                throw new InvalidOperationException($"Input must be a tensor type, got {Input.Type}");

            // Validate that starts, ends, and strides have the same length as input rank
            int rank = inputType.Shape.Length;
            if (Starts.Length != rank)
                throw new InvalidOperationException($"Starts length {Starts.Length} must match input rank {rank}");

            if (Ends.Length != rank)
                throw new InvalidOperationException($"Ends length {Ends.Length} must match input rank {rank}");

            if (Strides.Length != rank)
                throw new InvalidOperationException($"Strides length {Strides.Length} must match input rank {rank}");

            // Validate that strides are non-zero
            for (int i = 0; i < Strides.Length; i++)
            {
                if (Strides[i] == 0)
                    throw new InvalidOperationException($"Stride at dimension {i} cannot be zero");
            }

            // Validate that starts and ends are within bounds
            for (int i = 0; i < rank; i++)
            {
                int dimSize = inputType.Shape[i];
                if (Starts[i] < 0 || Starts[i] >= dimSize)
                    throw new InvalidOperationException($"Start {Starts[i]} at dimension {i} out of bounds [0, {dimSize})");

                if (Ends[i] < 0 || Ends[i] > dimSize)
                    throw new InvalidOperationException($"End {Ends[i]} at dimension {i} out of bounds [0, {dimSize}]");

                if (Starts[i] >= Ends[i] && Strides[i] > 0)
                    throw new InvalidOperationException($"Start {Starts[i]} must be less than end {Ends[i]} when stride is positive");
            }

            // Validate result type
            var expectedShape = ComputeOutputShape(inputType.Shape);
            var expectedType = new TensorType(inputType.ElementType, expectedShape);
            if (Result.Type != expectedType)
                throw new InvalidOperationException($"Result type mismatch: expected {expectedType}, got {Result.Type}");
        }

        /// <summary>
        /// Computes the output shape of the slice operation.
        /// </summary>
        private int[] ComputeOutputShape(int[] inputShape)
        {
            int[] outputShape = new int[inputShape.Length];
            for (int i = 0; i < inputShape.Length; i++)
            {
                int length = Ends[i] - Starts[i];
                outputShape[i] = (int)Math.Ceiling((double)length / Math.Abs(Strides[i]));
            }
            return outputShape;
        }

        /// <summary>
        /// Creates a clone of this operation.
        /// </summary>
        public override IROperation Clone()
        {
            return new SliceOp(Input, Result,
                             (int[])Starts.Clone(), (int[])Ends.Clone(), (int[])Strides.Clone());
        }
    }
}
