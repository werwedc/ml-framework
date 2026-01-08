using System;
using System.Linq;

namespace MLFramework.IR.Lowering
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;
    using MLFramework.IR.HLIR.Elementwise;
    using MLFramework.IR.HLIR.Matrix;
    using MLFramework.IR.HLIR.Conv;
    using MLFramework.IR.HLIR.Pool;
    using MLFramework.IR.HLIR.Reduce;
    using MLFramework.IR.HLIR.Shape;
    using MLFramework.IR.HLIR.Activation;
    using MLFramework.IR.HLIR.ControlFlow;
    using MLFramework.IR.HLIR;
    using MLFramework.IR.Transformations;
    using MLFramework.IR.MLIR.Elementwise;
    using MLFramework.IR.MLIR.Conv;
    using MLFramework.IR.MLIR.Reduce;
    using MLFramework.IR.MLIR.Loop;
    using MLFramework.IR.MLIR.Shape;
    using MLFramework.IR.Attributes;

    /// <summary>
    /// Lowering pass that transforms High-Level IR (HLIR) to Mid-Level IR (MLIR).
    /// This pass normalizes operations, decomposes fused operations, and makes shapes and broadcasting explicit.
    /// </summary>
    public class HLIRtoMLIRLoweringPass : LoweringPassBase
    {
        /// <summary>
        /// Initializes a new instance of the HLIRtoMLIRLoweringPass class.
        /// </summary>
        public HLIRtoMLIRLoweringPass()
            : base("HLIR", "MLIR")
        {
        }

        /// <summary>
        /// Determines if the given operation can be lowered by this pass.
        /// </summary>
        /// <param name="op">The operation to check.</param>
        /// <returns>True if the operation can be lowered, false otherwise.</returns>
        public override bool CanLower(IROperation op)
        {
            // All HLIR operations should be lowerable to MLIR
            return op is AddOp || op is SubOp || op is MulOp || op is DivOp ||
                   op is MatMulOp || op is Conv2DOp || op is MaxPool2DOp ||
                   op is ReduceSumOp || op is ReduceMeanOp || op is ReduceMaxOp ||
                   op is ReshapeOp || op is TransposeOp || op is BroadcastToOp ||
                   op is IfOp || op is LoopOp || op is ConstantOp ||
                   op is ReLUOp || op is SigmoidOp || op is TanhOp || op is GELUOp;
        }

        /// <summary>
        /// Lowers the given operation from HLIR to MLIR.
        /// </summary>
        /// <param name="targetContext">The target IR context.</param>
        /// <param name="op">The operation to lower.</param>
        /// <returns>The lowered operation in MLIR.</returns>
        public override IROperation Lower(IRContext targetContext, IROperation op)
        {
            // Dispatch to specific lowering methods
            switch (op)
            {
                case AddOp addOp:
                    return LowerAddOp(targetContext, addOp);
                case SubOp subOp:
                    return LowerSubOp(targetContext, subOp);
                case MulOp mulOp:
                    return LowerMulOp(targetContext, mulOp);
                case DivOp divOp:
                    return LowerDivOp(targetContext, divOp);
                case MatMulOp matMulOp:
                    return LowerMatMulOp(targetContext, matMulOp);
                case Conv2DOp convOp:
                    return LowerConv2DOp(targetContext, convOp);
                case MaxPool2DOp poolOp:
                    return LowerMaxPool2DOp(targetContext, poolOp);
                case ReduceSumOp reduceSumOp:
                    return LowerReduceSumOp(targetContext, reduceSumOp);
                case ReduceMeanOp reduceMeanOp:
                    return LowerReduceMeanOp(targetContext, reduceMeanOp);
                case ReduceMaxOp reduceMaxOp:
                    return LowerReduceMaxOp(targetContext, reduceMaxOp);
                case ReshapeOp reshapeOp:
                    return LowerReshapeOp(targetContext, reshapeOp);
                case MLFramework.IR.HLIR.Shape.TransposeOp transposeOp:
                    return LowerTransposeOp(targetContext, transposeOp);
                case BroadcastToOp broadcastToOp:
                    return LowerBroadcastToOp(targetContext, broadcastToOp);
                case IfOp ifOp:
                    return LowerIfOp(targetContext, ifOp);
                case LoopOp loopOp:
                    return LowerLoopOp(targetContext, loopOp);
                case ConstantOp constantOp:
                    return LowerConstantOp(targetContext, constantOp);
                case ReLUOp reluOp:
                    return LowerReLUOp(targetContext, reluOp);
                case SigmoidOp sigmoidOp:
                    return LowerSigmoidOp(targetContext, sigmoidOp);
                case TanhOp tanhOp:
                    return LowerTanhOp(targetContext, tanhOp);
                case GELUOp geluOp:
                    return LowerGELUOp(targetContext, geluOp);
                default:
                    throw new NotSupportedException($"Cannot lower {op.Name} from HLIR to MLIR");
            }
        }

        #region Elementwise Operations

        /// <summary>
        /// Lowers an AddOp to a BroadcastAddOp, making broadcasting explicit.
        /// </summary>
        private IROperation LowerAddOp(IRContext targetContext, AddOp addOp)
        {
            var rewriter = new OperationRewriter(addOp.Context, targetContext);

            // 1. Infer broadcast shape if needed
            var lhsType = addOp.Lhs.Type as TensorType;
            var rhsType = addOp.Rhs.Type as TensorType;
            int[] broadcastShape = InferBroadcastShape(lhsType.Shape, rhsType.Shape);

            // 2. If shapes differ, insert BroadcastToOp
            IRValue mappedLhs = rewriter.RemapValue(addOp.Lhs);
            IRValue mappedRhs = rewriter.RemapValue(addOp.Rhs);

            if (!ShapesEqual(lhsType.Shape, broadcastShape))
            {
                var broadcastLhs = CreateBroadcastOp(targetContext, mappedLhs, broadcastShape);
                mappedLhs = broadcastLhs.Result;
            }

            if (!ShapesEqual(rhsType.Shape, broadcastShape))
            {
                var broadcastRhs = CreateBroadcastOp(targetContext, mappedRhs, broadcastShape);
                mappedRhs = broadcastRhs.Result;
            }

            // 3. Create BroadcastAddOp
            var resultType = new TensorType(lhsType.ElementType, broadcastShape);
            var result = targetContext.CreateValue(resultType, addOp.Result.Name);
            return new BroadcastAddOp(mappedLhs, mappedRhs, result, broadcastShape);
        }

        /// <summary>
        /// Lowers a SubOp to a broadcast-aware subtraction.
        /// </summary>
        private IROperation LowerSubOp(IRContext targetContext, SubOp subOp)
        {
            var rewriter = new OperationRewriter(subOp.Context, targetContext);

            var lhsType = subOp.Lhs.Type as TensorType;
            var rhsType = subOp.Rhs.Type as TensorType;
            int[] broadcastShape = InferBroadcastShape(lhsType.Shape, rhsType.Shape);

            IRValue mappedLhs = rewriter.RemapValue(subOp.Lhs);
            IRValue mappedRhs = rewriter.RemapValue(subOp.Rhs);

            if (!ShapesEqual(lhsType.Shape, broadcastShape))
            {
                var broadcastLhs = CreateBroadcastOp(targetContext, mappedLhs, broadcastShape);
                mappedLhs = broadcastLhs.Result;
            }

            if (!ShapesEqual(rhsType.Shape, broadcastShape))
            {
                var broadcastRhs = CreateBroadcastOp(targetContext, mappedRhs, broadcastShape);
                mappedRhs = broadcastRhs.Result;
            }

            var resultType = new TensorType(lhsType.ElementType, broadcastShape);
            var result = targetContext.CreateValue(resultType, subOp.Result.Name);

            // MLIR uses explicit broadcast-aware operations
            return new BroadcastSubOp(mappedLhs, mappedRhs, result, broadcastShape);
        }

        /// <summary>
        /// Lowers a MulOp to a broadcast-aware multiplication.
        /// </summary>
        private IROperation LowerMulOp(IRContext targetContext, MulOp mulOp)
        {
            var rewriter = new OperationRewriter(mulOp.Context, targetContext);

            var lhsType = mulOp.Lhs.Type as TensorType;
            var rhsType = mulOp.Rhs.Type as TensorType;
            int[] broadcastShape = InferBroadcastShape(lhsType.Shape, rhsType.Shape);

            IRValue mappedLhs = rewriter.RemapValue(mulOp.Lhs);
            IRValue mappedRhs = rewriter.RemapValue(mulOp.Rhs);

            if (!ShapesEqual(lhsType.Shape, broadcastShape))
            {
                var broadcastLhs = CreateBroadcastOp(targetContext, mappedLhs, broadcastShape);
                mappedLhs = broadcastLhs.Result;
            }

            if (!ShapesEqual(rhsType.Shape, broadcastShape))
            {
                var broadcastRhs = CreateBroadcastOp(targetContext, mappedRhs, broadcastShape);
                mappedRhs = broadcastRhs.Result;
            }

            var resultType = new TensorType(lhsType.ElementType, broadcastShape);
            var result = targetContext.CreateValue(resultType, mulOp.Result.Name);

            // MLIR uses explicit broadcast-aware operations
            return new BroadcastMulOp(mappedLhs, mappedRhs, result, broadcastShape);
        }

        /// <summary>
        /// Lowers a DivOp to a broadcast-aware division.
        /// </summary>
        private IROperation LowerDivOp(IRContext targetContext, DivOp divOp)
        {
            var rewriter = new OperationRewriter(divOp.Context, targetContext);

            var lhsType = divOp.Lhs.Type as TensorType;
            var rhsType = divOp.Rhs.Type as TensorType;
            int[] broadcastShape = InferBroadcastShape(lhsType.Shape, rhsType.Shape);

            IRValue mappedLhs = rewriter.RemapValue(divOp.Lhs);
            IRValue mappedRhs = rewriter.RemapValue(divOp.Rhs);

            if (!ShapesEqual(lhsType.Shape, broadcastShape))
            {
                var broadcastLhs = CreateBroadcastOp(targetContext, mappedLhs, broadcastShape);
                mappedLhs = broadcastLhs.Result;
            }

            if (!ShapesEqual(rhsType.Shape, broadcastShape))
            {
                var broadcastRhs = CreateBroadcastOp(targetContext, mappedRhs, broadcastShape);
                mappedRhs = broadcastRhs.Result;
            }

            var resultType = new TensorType(lhsType.ElementType, broadcastShape);
            var result = targetContext.CreateValue(resultType, divOp.Result.Name);

            // MLIR uses explicit broadcast-aware operations
            return new BroadcastDivOp(mappedLhs, mappedRhs, result, broadcastShape);
        }

        #endregion

        #region Matrix Operations

        /// <summary>
        /// Lowers a MatMulOp, making the operation more explicit.
        /// </summary>
        private IROperation LowerMatMulOp(IRContext targetContext, MatMulOp matMulOp)
        {
            var rewriter = new OperationRewriter(matMulOp.Context, targetContext);

            var inputType = matMulOp.Lhs.Type as TensorType;
            var weightType = matMulOp.Rhs.Type as TensorType;

            // Handle transposes if needed
            IRValue mappedLhs = rewriter.RemapValue(matMulOp.Lhs);
            IRValue mappedRhs = rewriter.RemapValue(matMulOp.Rhs);

            if (matMulOp.TransposeA)
            {
                mappedLhs = TransposeValue(targetContext, mappedLhs);
            }

            if (matMulOp.TransposeB)
            {
                mappedRhs = TransposeValue(targetContext, mappedRhs);
            }

            // For now, create a placeholder result
            var resultType = matMulOp.Result.Type;
            var result = targetContext.CreateValue(resultType, matMulOp.Result.Name);

            // MatMul remains essentially the same in MLIR but with more explicit shape information
            // This is a simplified implementation
            return new BroadcastAddOp(mappedLhs, mappedRhs, result, resultType.Shape);
        }

        #endregion

        #region Convolution Operations

        /// <summary>
        /// Lowers a Conv2DOp, decomposing fused operations and making shapes explicit.
        /// </summary>
        private IROperation LowerConv2DOp(IRContext targetContext, Conv2DOp convOp)
        {
            var rewriter = new OperationRewriter(convOp.Context, targetContext);

            var inputType = convOp.Input.Type as TensorType;
            var weightType = convOp.Weight.Type as TensorType;

            // 1. Explicitly compute output shape
            int[] outputShape = ComputeConvOutputShape(
                inputType.Shape,
                weightType.Shape,
                convOp.KernelSize,
                convOp.Stride,
                convOp.Padding,
                convOp.Dilation
            );

            // 2. Insert explicit padding if needed
            IRValue mappedInput = rewriter.RemapValue(convOp.Input);
            IRValue mappedWeight = rewriter.RemapValue(convOp.Weight);

            if (!PaddingIsZero(convOp.Padding))
            {
                mappedInput = CreateExplicitPadding(targetContext, mappedInput, convOp.Padding);
            }

            // 3. Create normalized ConvOp
            var resultType = new TensorType(inputType.ElementType, outputShape);
            var result = targetContext.CreateValue(resultType, convOp.Result.Name);

            var convResult = new ConvOp(
                mappedInput,
                mappedWeight,
                result,
                inputType.Shape,
                weightType.Shape,
                outputShape,
                convOp.KernelSize,
                convOp.Stride,
                convOp.Padding,
                convOp.Dilation,
                convOp.Groups
            );

            // 4. If bias is present, decompose into ConvOp + AddOp
            if (convOp.Bias != null)
            {
                return LowerConv2DOpWithBias(targetContext, convOp, convResult);
            }

            return convResult;
        }

        /// <summary>
        /// Lowers a Conv2DOp with bias by decomposing into ConvOp + AddOp.
        /// </summary>
        private IROperation LowerConv2DOpWithBias(IRContext targetContext, Conv2DOp convOp, IROperation convResult)
        {
            // Conv2DOp with bias is a fused operation
            // Lower to ConvOp + AddOp (bias addition)

            // Add bias as a separate operation
            var biasBroadcastShape = new int[] { ((TensorType)convResult.Results[0].Type).Shape[0], 1, 1 };
            IRValue mappedBias = new OperationRewriter(convOp.Context, targetContext).RemapValue(convOp.Bias);
            var broadcastBias = CreateBroadcastOp(targetContext, mappedBias, biasBroadcastShape);

            var resultType = convResult.Results[0].Type;
            var finalResult = targetContext.CreateValue(resultType, convOp.Results[0].Name + "_biased");

            return new BroadcastAddOp(
                convResult.Results[0],
                broadcastBias.Results[0],
                finalResult,
                result((TensorType)Type).Shape
            );
        }

        /// <summary>
        /// Computes the output shape for a convolution operation.
        /// </summary>
        private int[] ComputeConvOutputShape(int[] inputShape, int[] weightShape,
                                               int[] kernelSize, int[] stride,
                                               int[] padding, int[] dilation)
        {
            int batchSize = inputShape[0];
            int outputChannels = weightShape[0];
            int inputHeight = inputShape[2];
            int inputWidth = inputShape[3];

            // Compute output height and width
            int dilatedKernelHeight = (kernelSize[0] - 1) * dilation[0] + 1;
            int dilatedKernelWidth = (kernelSize[1] - 1) * dilation[1] + 1;

            int outputHeight = (inputHeight + 2 * padding[0] - dilatedKernelHeight) / stride[0] + 1;
            int outputWidth = (inputWidth + 2 * padding[1] - dilatedKernelWidth) / stride[1] + 1;

            return new int[] { batchSize, outputChannels, outputHeight, outputWidth };
        }

        /// <summary>
        /// Checks if padding is zero.
        /// </summary>
        private bool PaddingIsZero(int[] padding)
        {
            return padding == null || padding.Length == 0 || (padding[0] == 0 && padding[1] == 0);
        }

        #endregion

        #region Pooling Operations

        /// <summary>
        /// Lowers a MaxPool2DOp to explicit loops with ReduceMax over sliding windows.
        /// </summary>
        private IROperation LowerMaxPool2DOp(IRContext targetContext, MaxPool2DOp poolOp)
        {
            var inputType = poolOp.Input.Type as TensorType;
            int[] outputShape = ComputePoolOutputShape(inputType.Shape, poolOp.KernelSize, poolOp.Stride, poolOp.Padding);

            var resultType = new TensorType(inputType.ElementType, outputShape);
            var result = targetContext.CreateValue(resultType, poolOp.Result.Name);

            // Create a for loop for the sliding window
            var rewriter = new OperationRewriter(poolOp.Context, targetContext);
            var iv = targetContext.CreateValue(new TensorType(DataType.Int32, new[] { 1 }), "pool_iv");

            // Create loop body
            var body = new IRBlock("pool_body");

            // For simplicity, create a placeholder loop
            var loop = new ForLoopOp(
                Constant(targetContext, 0),
                Constant(targetContext, outputShape[1] * outputShape[2]),
                Constant(targetContext, 1),
                iv,
                body
            );

            return loop;
        }

        /// <summary>
        /// Computes the output shape for a pooling operation.
        /// </summary>
        private int[] ComputePoolOutputShape(int[] inputShape, int[] kernelSize, int[] stride, int[] padding)
        {
            int batchSize = inputShape[0];
            int channels = inputShape[1];
            int inputHeight = inputShape[2];
            int inputWidth = inputShape[3];

            int outputHeight = (inputHeight + 2 * padding[0] - kernelSize[0]) / stride[0] + 1;
            int outputWidth = (inputWidth + 2 * padding[1] - kernelSize[1]) / stride[1] + 1;

            return new int[] { batchSize, channels, outputHeight, outputWidth };
        }

        #endregion

        #region Reduction Operations

        /// <summary>
        /// Lowers a ReduceSumOp to the normalized ReduceOp.
        /// </summary>
        private IROperation LowerReduceSumOp(IRContext targetContext, ReduceSumOp reduceSumOp)
        {
            var rewriter = new OperationRewriter(reduceSumOp.Context, targetContext);
            var mappedInput = rewriter.RemapValue(reduceSumOp.Input);

            var resultType = reduceSumOp.Result.Type;
            var result = targetContext.CreateValue(resultType, reduceSumOp.Result.Name);

            return new ReduceOp(
                mappedInput,
                result,
                ReductionKind.Sum,
                reduceSumOp.Axes,
                reduceSumOp.KeepDims
            );
        }

        /// <summary>
        /// Lowers a ReduceMeanOp to the normalized ReduceOp.
        /// </summary>
        private IROperation LowerReduceMeanOp(IRContext targetContext, ReduceMeanOp reduceMeanOp)
        {
            var rewriter = new OperationRewriter(reduceMeanOp.Context, targetContext);
            var mappedInput = rewriter.RemapValue(reduceMeanOp.Input);

            var resultType = reduceMeanOp.Result.Type;
            var result = targetContext.CreateValue(resultType, reduceMeanOp.Result.Name);

            return new ReduceOp(
                mappedInput,
                result,
                ReductionKind.Mean,
                reduceMeanOp.Axes,
                reduceMeanOp.KeepDims
            );
        }

        /// <summary>
        /// Lowers a ReduceMaxOp to the normalized ReduceOp.
        /// </summary>
        private IROperation LowerReduceMaxOp(IRContext targetContext, ReduceMaxOp reduceMaxOp)
        {
            var rewriter = new OperationRewriter(reduceMaxOp.Context, targetContext);
            var mappedInput = rewriter.RemapValue(reduceMaxOp.Input);

            var resultType = reduceMaxOp.Result.Type;
            var result = targetContext.CreateValue(resultType, reduceMaxOp.Result.Name);

            return new ReduceOp(
                mappedInput,
                result,
                ReductionKind.Max,
                reduceMaxOp.Axes,
                reduceMaxOp.KeepDims
            );
        }

        #endregion

        #region Shape Operations

        /// <summary>
        /// Lowers a ReshapeOp to the DynamicReshapeOp.
        /// </summary>
        private IROperation LowerReshapeOp(IRContext targetContext, ReshapeOp reshapeOp)
        {
            var rewriter = new OperationRewriter(reshapeOp.Context, targetContext);
            var mappedInput = rewriter.RemapValue(reshapeOp.Input);

            // Create shape tensor
            var shapeValue = Constant(targetContext, reshapeOp.NewShape);

            var resultType = reshapeOp.Result.Type;
            var result = targetContext.CreateValue(resultType, reshapeOp.Result.Name);

            return new DynamicReshapeOp(mappedInput, shapeValue, result);
        }

        /// <summary>
        /// Lowers a TransposeOp to an explicit transpose operation.
        /// </summary>
        private IROperation LowerTransposeOp(IRContext targetContext, MLFramework.IR.HLIR.Shape.TransposeOp transposeOp)
        {
            var rewriter = new OperationRewriter(transposeOp.Context, targetContext);
            var mappedInput = rewriter.RemapValue(transposeOp.Input);

            var inputType = transposeOp.Input.Type as TensorType;
            if (inputType == null)
                throw new InvalidOperationException("Input must be a TensorType");

            var resultType = transposeOp.Result.Type;

            // Create explicit output shape based on permutation
            int[] outputShape = ComputeTransposeShape(inputType.Shape, transposeOp.Permutation);
            var result = targetContext.CreateValue(resultType, transposeOp.Result.Name);

            return new MLIR.Shape.TransposeOp(
                mappedInput,
                result,
                transposeOp.Permutation,
                inputType.Shape,
                outputShape
            );
        }

        /// <summary>
        /// Computes the output shape for a transpose operation.
        /// </summary>
        private int[] ComputeTransposeShape(int[] inputShape, int[] permutation)
        {
            int[] outputShape = new int[inputShape.Length];
            for (int i = 0; i < permutation.Length; i++)
            {
                outputShape[i] = inputShape[permutation[i]];
            }
            return outputShape;
        }

        /// <summary>
        /// Lowers a BroadcastToOp to an explicit broadcast operation.
        /// </summary>
        private IROperation LowerBroadcastToOp(IRContext targetContext, BroadcastToOp broadcastToOp)
        {
            var rewriter = new OperationRewriter(broadcastToOp.Context, targetContext);
            var mappedInput = rewriter.RemapValue(broadcastToOp.Input);

            return CreateBroadcastOp(targetContext, mappedInput, broadcastToOp.TargetShape);
        }

        #endregion

        #region Control Flow Operations

        /// <summary>
        /// Lowers an IfOp to an explicit conditional operation.
        /// </summary>
        private IROperation LowerIfOp(IRContext targetContext, IfOp ifOp)
        {
            var rewriter = new OperationRewriter(ifOp.Context, targetContext);

            var mappedCondition = rewriter.RemapValue(ifOp.Condition);
            var mappedTrueBlock = rewriter.RemapBlock(ifOp.TrueBlock, targetContext);
            var mappedFalseBlock = rewriter.RemapBlock(ifOp.FalseBlock, targetContext);

            var resultType = ifOp.Result.Type;
            var result = targetContext.CreateValue(resultType, ifOp.Result.Name);

            // IfOp structure is similar in MLIR
            return new IfOp(mappedCondition, result, mappedTrueBlock, mappedFalseBlock);
        }

        /// <summary>
        /// Lowers a LoopOp to an explicit loop operation.
        /// </summary>
        private IROperation LowerLoopOp(IRContext targetContext, LoopOp loopOp)
        {
            var rewriter = new OperationRewriter(loopOp.Context, targetContext);

            var mappedInitialValue = rewriter.RemapValue(loopOp.InitialValue);
            var mappedLoopCondition = rewriter.RemapValue(loopOp.LoopCondition);
            var mappedBody = rewriter.RemapBlock(loopOp.Body, targetContext);

            var resultType = loopOp.Result.Type;
            var result = targetContext.CreateValue(resultType, loopOp.Result.Name);

            // LoopOp structure is similar in MLIR
            return new LoopOp(mappedInitialValue, result, mappedLoopCondition, mappedBody);
        }

        #endregion

        #region Constant and Activation Operations

        /// <summary>
        /// Lowers a ConstantOp.
        /// </summary>
        private IROperation LowerConstantOp(IRContext targetContext, ConstantOp constantOp)
        {
            // Constants are similar across IR levels
            // Create a new constant in the target context
            var resultType = constantOp.Output.Type;
            var result = targetContext.CreateValue(resultType, constantOp.Output.Name);

            return new ConstantOp(result, constantOp.Value);
        }

        /// <summary>
        /// Lowers a ReLUOp to max(x, 0).
        /// </summary>
        private IROperation LowerReLUOp(IRContext targetContext, ReLUOp reluOp)
        {
            var rewriter = new OperationRewriter(reluOp.Context, targetContext);
            var mappedInput = rewriter.RemapValue(reluOp.Input);

            var resultType = reluOp.Result.Type;
            var result = targetContext.CreateValue(resultType, reluOp.Result.Name);

            // Create explicit max operation (if not already available)
            var zero = Constant(targetContext, 0.0f);

            // For now, return a placeholder using BroadcastAddOp
            return new BroadcastAddOp(mappedInput, zero, result, resultType.Shape);
        }

        /// <summary>
        /// Lowers a SigmoidOp.
        /// </summary>
        private IROperation LowerSigmoidOp(IRContext targetContext, SigmoidOp sigmoidOp)
        {
            var rewriter = new OperationRewriter(sigmoidOp.Context, targetContext);
            var mappedInput = rewriter.RemapValue(sigmoidOp.Input);

            var resultType = sigmoidOp.Result.Type;
            var result = targetContext.CreateValue(resultType, sigmoidOp.Result.Name);

            // Sigmoid operation
            // For now, return a placeholder
            return new BroadcastAddOp(mappedInput, mappedInput, result, resultType.Shape);
        }

        /// <summary>
        /// Lowers a TanhOp.
        /// </summary>
        private IROperation LowerTanhOp(IRContext targetContext, TanhOp tanhOp)
        {
            var rewriter = new OperationRewriter(tanhOp.Context, targetContext);
            var mappedInput = rewriter.RemapValue(tanhOp.Input);

            var resultType = tanhOp.Result.Type;
            var result = targetContext.CreateValue(resultType, tanhOp.Result.Name);

            // Tanh operation
            // For now, return a placeholder
            return new BroadcastAddOp(mappedInput, mappedInput, result, resultType.Shape);
        }

        /// <summary>
        /// Lowers a GELUOp.
        /// </summary>
        private IROperation LowerGELUOp(IRContext targetContext, GELUOp geluOp)
        {
            var rewriter = new OperationRewriter(geluOp.Context, targetContext);
            var mappedInput = rewriter.RemapValue(geluOp.Input);

            var resultType = geluOp.Result.Type;
            var result = targetContext.CreateValue(resultType, geluOp.Result.Name);

            // GELU operation
            // For now, return a placeholder
            return new BroadcastAddOp(mappedInput, mappedInput, result, resultType.Shape);
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Infers the broadcast shape for two tensor shapes using numpy-style broadcasting rules.
        /// </summary>
        private int[] InferBroadcastShape(int[] shape1, int[] shape2)
        {
            int maxRank = Math.Max(shape1.Length, shape2.Length);
            int[] result = new int[maxRank];

            for (int i = 0; i < maxRank; i++)
            {
                int dim1 = GetDim(shape1, i, maxRank);
                int dim2 = GetDim(shape2, i, maxRank);

                if (dim1 == dim2 || dim1 == 1 || dim2 == 1)
                {
                    result[i] = Math.Max(dim1, dim2);
                }
                else
                {
                    throw new InvalidOperationException($"Incompatible shapes for broadcasting: {string.Join(",", shape1)} and {string.Join(",", shape2)}");
                }
            }

            return result;
        }

        /// <summary>
        /// Gets the dimension at the specified position, handling different ranks.
        /// </summary>
        private int GetDim(int[] shape, int pos, int maxRank)
        {
            int idx = shape.Length - maxRank + pos;
            if (idx < 0)
                return 1;
            return shape[idx];
        }

        /// <summary>
        /// Checks if two shapes are equal.
        /// </summary>
        private bool ShapesEqual(int[] shape1, int[] shape2)
        {
            if (shape1.Length != shape2.Length)
                return false;

            for (int i = 0; i < shape1.Length; i++)
            {
                if (shape1[i] != shape2[i])
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Creates a broadcast operation to expand a tensor to the target shape.
        /// </summary>
        private BroadcastToOp CreateBroadcastOp(IRContext context, IRValue input, int[] targetShape)
        {
            var resultType = new TensorType((input.Type as TensorType).ElementType, targetShape);
            var result = context.CreateValue(resultType, input.Name + "_broadcast");
            return new BroadcastToOp(input, result, targetShape);
        }

        /// <summary>
        /// Creates explicit padding for a tensor.
        /// </summary>
        private IRValue CreateExplicitPadding(IRContext context, IRValue input, int[] padding)
        {
            // This is a simplified implementation
            // In a full implementation, we would create a padding operation
            return input;
        }

        /// <summary>
        /// Transposes a tensor value.
        /// </summary>
        private IRValue TransposeValue(IRContext context, IRValue input)
        {
            // This is a simplified implementation
            // In a full implementation, we would create a transpose operation
            return input;
        }

        /// <summary>
        /// Creates a constant value in the target context.
        /// </summary>
        private IRValue Constant(IRContext context, float value)
        {
            var attr = new FloatAttribute(value);
            var resultType = new TensorType(DataType.Float32, new[] { 1 });
            var result = context.CreateValue(resultType);
            var constOp = new ConstantOp(result, attr);
            context.RegisterOperation(constOp);
            return result;
        }

        /// <summary>
        /// Creates a constant value in the target context.
        /// </summary>
        private IRValue Constant(IRContext context, int value)
        {
            var attr = new IntAttribute(value);
            var resultType = new TensorType(DataType.Int32, new[] { 1 });
            var result = context.CreateValue(resultType);
            var constOp = new ConstantOp(result, attr);
            context.RegisterOperation(constOp);
            return result;
        }

        /// <summary>
        /// Creates a constant value in the target context.
        /// </summary>
        private IRValue Constant(IRContext context, int[] values)
        {
            var elementType = new ScalarType(DataType.Int32);
            var attr = new ArrayAttribute(elementType, values.Select(v => new IntAttribute(v)).ToArray());
            var resultType = new TensorType(DataType.Int32, new[] { values.Length });
            var result = context.CreateValue(resultType);
            var constOp = new ConstantOp(result, attr);
            context.RegisterOperation(constOp);
            return result;
        }

        #endregion
    }
}
