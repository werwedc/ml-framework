using System;

namespace MLFramework.IR.HLIR.Shape
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;

    public class ReshapeOp : IROperation
    {
        public IRValue Input { get; }
        private IRValue _result;
        public IRValue Result => _result ?? Results[0];
        public int[] NewShape { get; }

        public ReshapeOp(IRValue input, IRValue result, int[] newShape)
            : base("reshape", IROpcode.Reshape, new[] { input }, new[] { result.Type }, null)
        {
            Input = input ?? throw new ArgumentNullException(nameof(input));
            _result = result ?? throw new ArgumentNullException(nameof(result));
            NewShape = newShape ?? throw new ArgumentNullException(nameof(newShape));
        }

        public override void Validate()
        {
            if (Input.Type is not TensorType inputType || Result.Type is not TensorType resultType)
                throw new InvalidOperationException("Input and result must be tensor types");

            int inputSize = 1;
            foreach (int dim in inputType.Shape)
                inputSize *= dim;

            int outputSize = 1;
            foreach (int dim in NewShape)
            {
                int actualDim = dim;
                if (dim == 0) actualDim = inputType.Shape[Array.IndexOf(NewShape, dim)];
                outputSize *= actualDim;
            }

            if (inputSize != outputSize)
                throw new InvalidOperationException($"Reshape size mismatch: {inputSize} vs {outputSize}");
        }

        public static IRValue Create(IRContext ctx, IRValue input, int[] newShape, string name = null)
        {
            var inputType = (TensorType)input.Type;
            var resultType = new TensorType(inputType.ElementType, newShape);
            var result = ctx.CreateValue(resultType, name);
            var op = new ReshapeOp(input, result, newShape);
            ctx.RegisterOperation(op);
            return result;
        }

        public override IROperation Clone() => new ReshapeOp(Input, Result, (int[])NewShape.Clone());
    }

    public class TransposeOp : IROperation
    {
        public IRValue Input { get; }
        private IRValue _result;
        public IRValue Result => _result ?? Results[0];
        public int[] Permutation { get; }

        public TransposeOp(IRValue input, IRValue result, int[] permutation)
            : base("transpose", IROpcode.Transpose, new[] { input }, new[] { result.Type }, null)
        {
            Input = input ?? throw new ArgumentNullException(nameof(input));
            _result = result ?? throw new ArgumentNullException(nameof(result));
            Permutation = permutation ?? throw new ArgumentNullException(nameof(permutation));
        }

        public override void Validate()
        {
            if (Input.Type is not TensorType inputType || Result.Type is not TensorType resultType)
                throw new InvalidOperationException("Input and result must be tensor types");
            if (Permutation.Length != inputType.Rank)
                throw new InvalidOperationException($"Permutation length {Permutation.Length} must match input rank {inputType.Rank}");

            foreach (int axis in Permutation)
                if (axis < 0 || axis >= inputType.Rank)
                    throw new InvalidOperationException($"Invalid axis {axis} for tensor with rank {inputType.Rank}");
        }

        public static IRValue Create(IRContext ctx, IRValue input, int[] permutation, string name = null)
        {
            var inputType = (TensorType)input.Type;
            int[] outputShape = new int[inputType.Rank];
            for (int i = 0; i < permutation.Length; i++)
                outputShape[i] = inputType.Shape[permutation[i]];
            var resultType = new TensorType(inputType.ElementType, outputShape);
            var result = ctx.CreateValue(resultType, name);
            var op = new TransposeOp(input, result, permutation);
            ctx.RegisterOperation(op);
            return result;
        }

        public override IROperation Clone() => new TransposeOp(Input, Result, (int[])Permutation.Clone());
    }

    public class BroadcastToOp : IROperation
    {
        public IRValue Input { get; }
        private IRValue _result;
        public IRValue Result => _result ?? Results[0];
        public int[] TargetShape { get; }

        public BroadcastToOp(IRValue input, IRValue result, int[] targetShape)
            : base("broadcast_to", IROpcode.BroadcastTo, new[] { input }, new[] { result.Type }, null)
        {
            Input = input ?? throw new ArgumentNullException(nameof(input));
            _result = result ?? throw new ArgumentNullException(nameof(result));
            TargetShape = targetShape ?? throw new ArgumentNullException(nameof(targetShape));
        }

        public override void Validate()
        {
            if (Input.Type is not TensorType inputType || Result.Type is not TensorType resultType)
                throw new InvalidOperationException("Input and result must be tensor types");

            int maxRank = Math.Max(inputType.Rank, TargetShape.Length);
            for (int i = 0; i < maxRank; i++)
            {
                int dim1 = i < inputType.Rank ? inputType.Shape[inputType.Rank - maxRank + i] : 1;
                int dim2 = TargetShape[TargetShape.Length - maxRank + i];

                if (dim1 != dim2 && dim1 != 1)
                    throw new InvalidOperationException($"Cannot broadcast shape {string.Join(",", inputType.Shape)} to {string.Join(",", TargetShape)}");
            }
        }

        public static IRValue Create(IRContext ctx, IRValue input, int[] targetShape, string name = null)
        {
            var inputType = (TensorType)input.Type;
            var resultType = new TensorType(inputType.ElementType, targetShape);
            var result = ctx.CreateValue(resultType, name);
            var op = new BroadcastToOp(input, result, targetShape);
            ctx.RegisterOperation(op);
            return result;
        }

        public override IROperation Clone() => new BroadcastToOp(Input, Result, (int[])TargetShape.Clone());
    }
}
