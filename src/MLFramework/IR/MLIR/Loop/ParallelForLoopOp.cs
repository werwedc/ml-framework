using System;
using MLFramework.IR.Operations;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.IR.MLIR.Loop
{
    /// <summary>
    /// Parallel for loop operation in MLIR.
    /// Similar to ForLoopOp but marked for parallel execution.
    /// </summary>
    public class ParallelForLoopOp : IROperation
    {
        /// <summary>Gets the lower bound of the loop.</summary>
        public IRValue LowerBound => Operands[0];

        /// <summary>Gets the upper bound of the loop.</summary>
        public IRValue UpperBound => Operands[1];

        /// <summary>Gets the step size of the loop.</summary>
        public IRValue Step => Operands[2];

        /// <summary>Gets the induction variable of the loop.</summary>
        public IRValue InductionVariable => Results[0];

        /// <summary>Gets the loop body block.</summary>
        public IRBlock Body { get; }

        /// <summary>
        /// Gets the number of threads for parallel execution.
        /// </summary>
        public int NumThreads { get; }

        /// <summary>
        /// Initializes a new instance of the ParallelForLoopOp class.
        /// </summary>
        /// <param name="lowerBound">The lower bound (inclusive).</param>
        /// <param name="upperBound">The upper bound (exclusive).</param>
        /// <param name="step">The step size.</param>
        /// <param name="inductionVariable">The induction variable result.</param>
        /// <param name="body">The loop body block.</param>
        /// <param name="numThreads">The number of threads for parallel execution.</param>
        public ParallelForLoopOp(IRValue lowerBound, IRValue upperBound, IRValue step,
                                IRValue inductionVariable, IRBlock body, int numThreads)
            : base("parallel_for", IROpcode.ParallelForLoopOp, new[] { lowerBound, upperBound, step }, new[] { inductionVariable.Type }, null)
        {
            Results[0] = inductionVariable ?? throw new ArgumentNullException(nameof(inductionVariable));
            Body = body ?? throw new ArgumentNullException(nameof(body));
            NumThreads = numThreads;

            if (numThreads <= 0)
                throw new ArgumentException("NumThreads must be positive", nameof(numThreads));
        }

        /// <summary>
        /// Validates the parallel for loop operation.
        /// </summary>
        public override void Validate()
        {
            // Validate that bounds and step are scalars (or 1D tensors with size 1)
            ValidateScalar(LowerBound, "LowerBound");
            ValidateScalar(UpperBound, "UpperBound");
            ValidateScalar(Step, "Step");

            // Validate that induction variable is a scalar
            ValidateScalar(InductionVariable, "InductionVariable");

            // Validate that bounds and step have the same type
            if (LowerBound.Type != UpperBound.Type || LowerBound.Type != Step.Type)
                throw new InvalidOperationException(
                    $"Lower bound, upper bound, and step must have the same type");

            // Validate that induction variable type matches bounds type
            if (InductionVariable.Type != LowerBound.Type)
                throw new InvalidOperationException(
                    $"Induction variable type must match bound type");

            // Validate body block
            if (Body == null)
                throw new InvalidOperationException("Body block cannot be null");

            // Validate thread count
            if (NumThreads <= 0)
                throw new InvalidOperationException($"NumThreads must be positive, got {NumThreads}");
        }

        /// <summary>
        /// Validates that a value is a scalar (or 1D tensor with size 1).
        /// </summary>
        private void ValidateScalar(IRValue value, string name)
        {
            if (!(value.Type is TensorType tensorType))
                throw new InvalidOperationException($"{name} must be a tensor type, got {value.Type}");

            if (tensorType.Shape.Length > 1 || (tensorType.Shape.Length == 1 && tensorType.Shape[0] != 1))
                throw new InvalidOperationException($"{name} must be a scalar (0D or 1D with size 1)");
        }

        /// <summary>
        /// Creates a clone of this operation.
        /// </summary>
        public override IROperation Clone()
        {
            // Note: Body block cloning would need to be implemented separately
            return new ParallelForLoopOp(LowerBound, UpperBound, Step, InductionVariable, Body, NumThreads);
        }
    }
}
