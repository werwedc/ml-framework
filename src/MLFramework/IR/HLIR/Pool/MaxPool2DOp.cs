using System;

namespace MLFramework.IR.HLIR.Pool
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;

    public class MaxPool2DOp : IROperation
    {
        public IRValue Input { get; }
        private IRValue _result;
        public IRValue Result => _result ?? Results[0];
        public int[] KernelSize { get; }
        public int[] Stride { get; }
        public int[] Padding { get; }

        public MaxPool2DOp(IRValue input, IRValue result, int[] kernelSize, int[] stride, int[] padding = null)
            : base("maxpool2d", IROpcode.MaxPool2D, new[] { input }, new[] { result.Type }, null)
        {
            Input = input ?? throw new ArgumentNullException(nameof(input));
            _result = result ?? throw new ArgumentNullException(nameof(result));
            KernelSize = kernelSize ?? throw new ArgumentNullException(nameof(kernelSize));
            Stride = stride ?? throw new ArgumentNullException(nameof(stride));
            Padding = padding ?? new int[] { 0, 0 };
            Results[0] = result;
        }

        public override void Validate()
        {
            if (Input.Type is not TensorType inputType || Result.Type is not TensorType resultType)
                throw new InvalidOperationException("Input and result must be tensor types");
            if (inputType.Shape.Length != 4)
                throw new InvalidOperationException("Input must be 4D tensor (NCHW format)");
            if (KernelSize.Length != 2 || Stride.Length != 2 || Padding.Length != 2)
                throw new InvalidOperationException("KernelSize, Stride, and Padding must be 2-element arrays");
        }

        public static IRValue Create(IRContext ctx, IRValue input, int[] kernelSize, int[] stride, int[] padding = null, string name = null)
        {
            var inputType = (TensorType)input.Type;
            int[] outputShape = ComputeOutputShape(inputType.Shape, kernelSize, stride, padding);
            var resultType = new TensorType(inputType.ElementType, outputShape);
            var result = ctx.CreateValue(resultType, name);
            var op = new MaxPool2DOp(input, result, kernelSize, stride, padding);
            ctx.RegisterOperation(op);
            return result;
        }

        private static int[] ComputeOutputShape(int[] inputShape, int[] kernelSize, int[] stride, int[] padding)
        {
            int outputHeight = (inputShape[2] + 2 * padding[0] - kernelSize[0]) / stride[0] + 1;
            int outputWidth = (inputShape[3] + 2 * padding[1] - kernelSize[1]) / stride[1] + 1;
            return new int[] { inputShape[0], inputShape[1], outputHeight, outputWidth };
        }

        public override IROperation Clone() => new MaxPool2DOp(Input, Result, KernelSize, Stride, Padding);
    }

    public class AvgPool2DOp : IROperation
    {
        public IRValue Input { get; }
        private IRValue _result;
        public IRValue Result => _result ?? Results[0];
        public int[] KernelSize { get; }
        public int[] Stride { get; }
        public int[] Padding { get; }

        public AvgPool2DOp(IRValue input, IRValue result, int[] kernelSize, int[] stride, int[] padding = null)
            : base("avgpool2d", IROpcode.AvgPool2D, new[] { input }, new[] { result.Type }, null)
        {
            Input = input ?? throw new ArgumentNullException(nameof(input));
            _result = result ?? throw new ArgumentNullException(nameof(result));
            KernelSize = kernelSize ?? throw new ArgumentNullException(nameof(kernelSize));
            Stride = stride ?? throw new ArgumentNullException(nameof(stride));
            Padding = padding ?? new int[] { 0, 0 };
        }

        public override void Validate()
        {
            if (Input.Type is not TensorType inputType || Result.Type is not TensorType resultType)
                throw new InvalidOperationException("Input and result must be tensor types");
            if (inputType.Shape.Length != 4)
                throw new InvalidOperationException("Input must be 4D tensor (NCHW format)");
            if (KernelSize.Length != 2 || Stride.Length != 2 || Padding.Length != 2)
                throw new InvalidOperationException("KernelSize, Stride, and Padding must be 2-element arrays");
        }

        public static IRValue Create(IRContext ctx, IRValue input, int[] kernelSize, int[] stride, int[] padding = null, string name = null)
        {
            var inputType = (TensorType)input.Type;
            int[] outputShape = ComputeOutputShape(inputType.Shape, kernelSize, stride, padding);
            var resultType = new TensorType(inputType.ElementType, outputShape);
            var result = ctx.CreateValue(resultType, name);
            var op = new AvgPool2DOp(input, result, kernelSize, stride, padding);
            ctx.RegisterOperation(op);
            return result;
        }

        private static int[] ComputeOutputShape(int[] inputShape, int[] kernelSize, int[] stride, int[] padding)
        {
            int outputHeight = (inputShape[2] + 2 * padding[0] - kernelSize[0]) / stride[0] + 1;
            int outputWidth = (inputShape[3] + 2 * padding[1] - kernelSize[1]) / stride[1] + 1;
            return new int[] { inputShape[0], inputShape[1], outputHeight, outputWidth };
        }

        public override IROperation Clone() => new AvgPool2DOp(Input, Result, KernelSize, Stride, Padding);
    }
}
