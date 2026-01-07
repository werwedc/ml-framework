using System;
using System.Linq;

namespace MLFramework.IR.HLIR.Reduce
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;

    public class ReduceSumOp : IROperation
    {
        public IRValue Input { get; }
        private IRValue _result;
        public IRValue Result => _result ?? Results[0];
        public int[] Axes { get; }
        public bool KeepDims { get; }

        public ReduceSumOp(IRValue input, IRValue result, int[] axes, bool keepDims = false)
            : base("reduce_sum", IROpcode.ReduceSum, new[] { input }, new[] { result.Type }, null)
        {
            Input = input ?? throw new ArgumentNullException(nameof(input));
            _result = result ?? throw new ArgumentNullException(nameof(result));
            Axes = axes ?? throw new ArgumentNullException(nameof(axes));
            KeepDims = keepDims;
        }

        public override void Validate()
        {
            if (Input.Type is not TensorType inputType || Result.Type is not TensorType resultType)
                throw new InvalidOperationException("Input and result must be tensor types");
            foreach (int axis in Axes)
                if (axis < -inputType.Rank || axis >= inputType.Rank)
                    throw new InvalidOperationException($"Invalid axis {axis} for tensor with rank {inputType.Rank}");
        }

        public static IRValue Create(IRContext ctx, IRValue input, int[] axes, bool keepDims = false, string name = null)
        {
            var inputType = (TensorType)input.Type;
            int[] outputShape = ComputeOutputShape(inputType.Shape, axes, keepDims);
            var resultType = new TensorType(inputType.ElementType, outputShape);
            var result = ctx.CreateValue(resultType, name);
            var op = new ReduceSumOp(input, result, axes, keepDims);
            ctx.RegisterOperation(op);
            return result;
        }

        public static int[] ComputeOutputShape(int[] inputShape, int[] axes, bool keepDims)
        {
            int[] normalizedAxes = axes.Select(a => a < 0 ? inputShape.Length + a : a).Distinct().ToArray();
            int[] outputShape = (int[])inputShape.Clone();

            foreach (int axis in normalizedAxes.OrderByDescending(x => x))
                if (keepDims)
                    outputShape[axis] = 1;
                else
                    outputShape = outputShape.Where((_, i) => i != axis).ToArray();

            return outputShape;
        }

        public override IROperation Clone() => new ReduceSumOp(Input, Result, Axes, KeepDims);
    }

    public class ReduceMeanOp : IROperation
    {
        public IRValue Input { get; }
        private IRValue _result;
        public IRValue Result => _result ?? Results[0];
        public int[] Axes { get; }
        public bool KeepDims { get; }

        public ReduceMeanOp(IRValue input, IRValue result, int[] axes, bool keepDims = false)
            : base("reduce_mean", IROpcode.ReduceMean, new[] { input }, new[] { result.Type }, null)
        {
            Input = input ?? throw new ArgumentNullException(nameof(input));
            _result = result ?? throw new ArgumentNullException(nameof(result));
            Axes = axes ?? throw new ArgumentNullException(nameof(axes));
            KeepDims = keepDims;
        }

        public override void Validate()
        {
            if (Input.Type is not TensorType inputType || Result.Type is not TensorType resultType)
                throw new InvalidOperationException("Input and result must be tensor types");
            foreach (int axis in Axes)
                if (axis < -inputType.Rank || axis >= inputType.Rank)
                    throw new InvalidOperationException($"Invalid axis {axis} for tensor with rank {inputType.Rank}");
        }

        public static IRValue Create(IRContext ctx, IRValue input, int[] axes, bool keepDims = false, string name = null)
        {
            var inputType = (TensorType)input.Type;
            int[] outputShape = ReduceSumOp.ComputeOutputShape(inputType.Shape, axes, keepDims);
            var resultType = new TensorType(inputType.ElementType, outputShape);
            var result = ctx.CreateValue(resultType, name);
            var op = new ReduceMeanOp(input, result, axes, keepDims);
            ctx.RegisterOperation(op);
            return result;
        }

        public override IROperation Clone() => new ReduceMeanOp(Input, Result, Axes, KeepDims);
    }

    public class ReduceMaxOp : IROperation
    {
        public IRValue Input { get; }
        private IRValue _result;
        public IRValue Result => _result ?? Results[0];
        public int[] Axes { get; }
        public bool KeepDims { get; }

        public ReduceMaxOp(IRValue input, IRValue result, int[] axes, bool keepDims = false)
            : base("reduce_max", IROpcode.ReduceMax, new[] { input }, new[] { result.Type }, null)
        {
            Input = input ?? throw new ArgumentNullException(nameof(input));
            _result = result ?? throw new ArgumentNullException(nameof(result));
            Axes = axes ?? throw new ArgumentNullException(nameof(axes));
            KeepDims = keepDims;
        }

        public override void Validate()
        {
            if (Input.Type is not TensorType inputType || Result.Type is not TensorType resultType)
                throw new InvalidOperationException("Input and result must be tensor types");
            foreach (int axis in Axes)
                if (axis < -inputType.Rank || axis >= inputType.Rank)
                    throw new InvalidOperationException($"Invalid axis {axis} for tensor with rank {inputType.Rank}");
        }

        public static IRValue Create(IRContext ctx, IRValue input, int[] axes, bool keepDims = false, string name = null)
        {
            var inputType = (TensorType)input.Type;
            int[] outputShape = ReduceSumOp.ComputeOutputShape(inputType.Shape, axes, keepDims);
            var resultType = new TensorType(inputType.ElementType, outputShape);
            var result = ctx.CreateValue(resultType, name);
            var op = new ReduceMaxOp(input, result, axes, keepDims);
            ctx.RegisterOperation(op);
            return result;
        }

        public override IROperation Clone() => new ReduceMaxOp(Input, Result, Axes, KeepDims);
    }
}
