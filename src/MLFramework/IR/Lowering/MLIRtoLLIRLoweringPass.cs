using System;
using MLFramework.IR.Backend;
using MLFramework.IR.Operations;
using MLFramework.IR.LLIR.Operations.Arithmetic;
using MLFramework.IR.LLIR.Operations.ControlFlow;
using MLFramework.IR.LLIR.Operations.Memory;
using MLFramework.IR.LLIR.Types;
using MLFramework.IR.LLIR.Values;
using MLFramework.IR.MLIR.Conv;
using MLFramework.IR.MLIR.Elementwise;
using MLFramework.IR.MLIR.Loop;
using MLFramework.IR.MLIR.Memory;
using MLFramework.IR.MLIR.Reduce;
using MLFramework.IR.Transformations;
using MLFramework.IR.Types;
using MLFramework.IR.Values;
using IRBlock = MLFramework.IR.IRBlock;

namespace MLFramework.IR.Lowering
{
    /// <summary>
    /// Lowering pass that transforms Mid-Level IR (MLIR) to Low-Level IR (LLIR).
    /// This pass makes memory operations explicit, introduces registers and buffers,
    /// and converts high-level operations to low-level arithmetic.
    /// </summary>
    public class MLIRtoLLIRLoweringPass : LoweringPassBase
    {
        private MemoryLayout _memoryLayout;
        private int _vectorWidth;

        /// <summary>
        /// Initializes a new instance of the MLIRtoLLIRLoweringPass class.
        /// </summary>
        /// <param name="memoryLayout">The memory layout (row-major or column-major).</param>
        /// <param name="vectorWidth">The vector width for SIMD operations (1 for scalar).</param>
        public MLIRtoLLIRLoweringPass(MemoryLayout memoryLayout = MemoryLayout.RowMajor,
                                      int vectorWidth = 1)
            : base("MLIR", "LLIR")
        {
            _memoryLayout = memoryLayout;
            _vectorWidth = vectorWidth;
        }

        /// <summary>
        /// Determines if given operation can be lowered to LLIR.
        /// </summary>
        /// <param name="op">The operation to check.</param>
        /// <returns>True if operation can be lowered, false otherwise.</returns>
        public override bool CanLower(IROperation op)
        {
            // All MLIR operations should be lowerable to LLIR
            return op is BroadcastAddOp || op is ConvOp || op is AllocTensorOp ||
                   op is ForLoopOp || op is ReduceOp;
        }

        /// <summary>
        /// Lowers the given operation from MLIR to LLIR.
        /// </summary>
        /// <param name="targetContext">The target IR context.</param>
        /// <param name="op">The operation to lower.</param>
        /// <returns>The lowered operation in LLIR.</returns>
        public override IROperation Lower(IRContext targetContext, IROperation op)
        {
            switch (op)
            {
                case BroadcastAddOp addOp: return LowerBroadcastAddOp(targetContext, addOp);
                case ConvOp convOp: return LowerConvOp(targetContext, convOp);
                case AllocTensorOp allocOp: return LowerAllocTensorOp(targetContext, allocOp);
                case ForLoopOp loopOp: return LowerForLoopOp(targetContext, loopOp);
                case ReduceOp reduceOp: return LowerReduceOp(targetContext, reduceOp);
                default: throw new NotSupportedException($"Cannot lower {op.Name}");
            }
        }

        #region Memory Allocation Lowering

        /// <summary>
        /// Lowers an AllocTensorOp to LLIR memory operations.
        /// </summary>
        private IROperation LowerAllocTensorOp(IRContext targetContext, AllocTensorOp allocOp)
        {
            var tensorType = allocOp.AllocatedType;
            int sizeInBytes = ComputeTensorSize(tensorType);
            int alignment = ComputeAlignment(tensorType.ElementType);

            var buffer = new MemoryValue(
                new PointerType(new ScalarType(tensorType.ElementType)),
                $"{allocOp.Result.Name}_buffer",
                0,  // Offset computed by memory allocator
                sizeInBytes
            );

            var bufferAllocOp = new AllocBufferOp(buffer, sizeInBytes, alignment);
            targetContext.RegisterOperation(bufferAllocOp);

            return bufferAllocOp;
        }

        /// <summary>
        /// Computes the size of a tensor in bytes.
        /// </summary>
        private int ComputeTensorSize(TensorType type)
        {
            int totalElements = 1;
            foreach (int dim in type.Shape)
            {
                if (dim <= 0)  // Dynamic dimension
                    throw new NotSupportedException("Cannot compute size of tensor with dynamic shape");
                totalElements *= dim;
            }

            return totalElements * GetElementSize(type.ElementType);
        }

        /// <summary>
        /// Gets the size of an element type in bytes.
        /// </summary>
        private int GetElementSize(DataType type)
        {
            return type switch
            {
                DataType.Float32 or DataType.Int32 or DataType.UInt32 => 4,
                DataType.Float64 or DataType.Int64 or DataType.UInt64 => 8,
                DataType.Float16 => 2,
                DataType.Int8 or DataType.UInt8 or DataType.Bool => 1,
                _ => throw new ArgumentException($"Unsupported element type: {type}")
            };
        }

        /// <summary>
        /// Computes the alignment requirement for a given element type.
        /// </summary>
        private int ComputeAlignment(DataType elementType)
        {
            // Align to element type size, minimum 16 bytes for SIMD
            return Math.Max(GetElementSize(elementType), 16);
        }

        #endregion

        #region Elementwise Operation Lowering

        /// <summary>
        /// Lowers a BroadcastAddOp to LLIR loop operations.
        /// </summary>
        private IROperation LowerBroadcastAddOp(IRContext targetContext, BroadcastAddOp addOp)
        {
            var rewriter = new OperationRewriter(addOp.Context, targetContext);

            var lhsType = addOp.Lhs.Type as TensorType;
            var rhsType = addOp.Rhs.Type as TensorType;
            var resultType = addOp.Result.Type as TensorType;

            // Allocate register for result
            var resultRegister = new RegisterValue(resultType, $"{addOp.Result.Name}_reg");

            // Generate loops for each dimension
            var resultBuffer = rewriter.RemapValue(addOp.Result) as MemoryValue;

            return GenerateElementwiseLoop(
                targetContext,
                rewriter.RemapValue(addOp.Lhs) as MemoryValue,
                rewriter.RemapValue(addOp.Rhs) as MemoryValue,
                resultBuffer,
                 resultType.Shape,
                 (lhsAddr, rhsAddr, resultAddr) => new AddScalarOp(
                     new RegisterValue(new ScalarType(lhsType.ElementType), "lhs_val"),
                     new RegisterValue(new ScalarType(rhsType.ElementType), "rhs_val"),
                     resultRegister
                 )
            );
        }

        /// <summary>
        /// Generates elementwise loops for tensor operations.
        /// </summary>
        private IROperation GenerateElementwiseLoop(IRContext targetContext,
                                                  MemoryValue lhs,
                                                  MemoryValue rhs,
                                                  MemoryValue result,
                                                  int[] shape,
                                                  Func<LLIRValue, LLIRValue, LLIRValue, IROperation> bodyBuilder)
        {
            // Generate nested loops for multi-dimensional tensors
            // Each loop iterates over one dimension
            // Loop body performs the operation

            if (shape.Length == 1)
            {
                // 1D tensor - single loop
                return Generate1DLoop(targetContext, lhs, rhs, result, shape[0], bodyBuilder);
            }
            else
            {
                // Multi-dimensional tensor - nested loops
                return GenerateNDLoop(targetContext, lhs, rhs, result, shape, bodyBuilder);
            }
        }

        /// <summary>
        /// Generates a 1D loop for elementwise operations.
        /// </summary>
        private IROperation Generate1DLoop(IRContext targetContext,
                                           MemoryValue lhs,
                                           MemoryValue rhs,
                                           MemoryValue result,
                                           int size,
                                           Func<LLIRValue, LLIRValue, LLIRValue, IROperation> bodyBuilder)
        {
            var bodyBlock = new IRBlock("elemwise_body");
            var iv = new RegisterValue(new ScalarType(DataType.Int32), "i");
            var start = Constant(0);
            var end = Constant(size);
            var step = Constant(1);

            // In loop body:
            // 1. Compute addresses: lhs + i * elementSize
            // 2. Load values
            // 3. Perform operation
            // 4. Store result

            var lhsAddr = ComputeAddress(lhs, iv);
            var rhsAddr = ComputeAddress(rhs, iv);
            var resultAddr = ComputeAddress(result, iv);

            var lhsVal = new RegisterValue(new ScalarType(DataType.Float32), "lhs_val");
            var rhsVal = new RegisterValue(new ScalarType(DataType.Float32), "rhs_val");
            var resultVal = new RegisterValue(new ScalarType(DataType.Float32), "result_val");

            bodyBlock.AddOperation(new LoadOp(lhsAddr, lhsVal));
            bodyBlock.AddOperation(new LoadOp(rhsAddr, rhsVal));
            bodyBlock.AddOperation(bodyBuilder(lhsVal, rhsVal, resultVal));
            bodyBlock.AddOperation(new StoreOp(resultAddr, resultVal));

            return new LLIRForLoopOp(start, end, step, iv, bodyBlock);
        }

        /// <summary>
        /// Generates an ND loop for elementwise operations.
        /// </summary>
        private IROperation GenerateNDLoop(IRContext targetContext,
                                          MemoryValue lhs,
                                          MemoryValue rhs,
                                          MemoryValue result,
                                          int[] shape,
                                          Func<LLIRValue, LLIRValue, LLIRValue, IROperation> bodyBuilder)
        {
            // Simplified implementation - in a full version, we would recursively
            // generate nested loops for each dimension
            return Generate1DLoop(targetContext, lhs, rhs, result, shape[0], bodyBuilder);
        }

        #endregion

        #region Convolution Lowering

        /// <summary>
        /// Lowers a ConvOp to LLIR loop operations.
        /// </summary>
        private IROperation LowerConvOp(IRContext targetContext, ConvOp convOp)
        {
            var rewriter = new OperationRewriter(convOp.Context, targetContext);

            var input = rewriter.RemapValue(convOp.Input) as MemoryValue;
            var weight = rewriter.RemapValue(convOp.Weight) as MemoryValue;
            var result = rewriter.RemapValue(convOp.Result) as MemoryValue;

            // Convolution lowering is complex:
            // 1. Generate nested loops for batch, output height, output width, output channels
            // 2. Inner loops iterate over kernel height, kernel width, input channels
            // 3. Accumulate convolution sum in a register
            // 4. Store result to output

            var bodyBlock = new IRBlock("conv_body");
            var batchSize = convOp.InputShape[0];
            var outputHeight = convOp.OutputShape[1];
            var outputWidth = convOp.OutputShape[2];
            var outputChannels = convOp.OutputShape[3];

            // Create induction variables
            var batchIV = new RegisterValue(new ScalarType(DataType.Int32), "b");
            var ohIV = new RegisterValue(new ScalarType(DataType.Int32), "oh");
            var owIV = new RegisterValue(new ScalarType(DataType.Int32), "ow");
            var ocIV = new RegisterValue(new ScalarType(DataType.Int32), "oc");

            // Accumulator register
            var accumulator = new RegisterValue(new ScalarType(DataType.Float32), "acc");
            bodyBlock.AddOperation(new AddScalarOp(accumulator, Constant(0.0f), accumulator));

            // Inner loops for kernel
            var kernelHeightIV = new RegisterValue(new ScalarType(DataType.Int32), "kh");
            var kernelWidthIV = new RegisterValue(new ScalarType(DataType.Int32), "kw");
            var inputChannelsIV = new RegisterValue(new ScalarType(DataType.Int32), "ic");

            // Generate complex nested loop structure
            // (simplified - full implementation would generate all nested loops)
            // Returning null as this is a placeholder for complex convolution lowering

            return null;
        }

        #endregion

        #region Reduction Lowering

        /// <summary>
        /// Lowers a ReduceOp to LLIR loop operations.
        /// </summary>
        private IROperation LowerReduceOp(IRContext targetContext, ReduceOp reduceOp)
        {
            var rewriter = new OperationRewriter(reduceOp.Context, targetContext);

            var input = rewriter.RemapValue(reduceOp.Input) as MemoryValue;
            var result = rewriter.RemapValue(reduceOp.Result) as MemoryValue;

            // Reduction lowering:
            // 1. Initialize result to identity value
            // 2. Loop over reduction axes
            // 3. Apply reduction operation
            // 4. Handle keep_dims if necessary

            var reductionOp = GetReductionOperation(reduceOp.Kind);
            var identityValue = GetIdentityValue(reduceOp.Kind, input.Type);

            return GenerateReductionLoop(targetContext, input, result, reduceOp.Axes,
                                         reduceOp.KeepDims, reductionOp, identityValue);
        }

        /// <summary>
        /// Gets the reduction operation for a given reduction kind.
        /// </summary>
        private Func<LLIRValue, LLIRValue, LLIRValue> GetReductionOperation(ReductionKind kind)
        {
            return kind switch
            {
                ReductionKind.Sum => (a, b) => new RegisterValue(a.Type, "sum"),
                ReductionKind.Max => (a, b) => new RegisterValue(a.Type, "max"),
                ReductionKind.Min => (a, b) => new RegisterValue(a.Type, "min"),
                _ => throw new NotSupportedException($"Reduction kind {kind} not yet supported")
            };
        }

        /// <summary>
        /// Gets the identity value for a given reduction kind and type.
        /// </summary>
        private LLIRValue GetIdentityValue(ReductionKind kind, IIRType type)
        {
            return kind switch
            {
                ReductionKind.Sum => new RegisterValue(type, "0"),
                ReductionKind.Max => new RegisterValue(type, "neg_inf"),
                ReductionKind.Min => new RegisterValue(type, "pos_inf"),
                _ => new RegisterValue(type, "identity")
            };
        }

        /// <summary>
        /// Generates a reduction loop.
        /// </summary>
        private IROperation GenerateReductionLoop(IRContext targetContext,
                                                 MemoryValue input,
                                                 MemoryValue result,
                                                 int[] axes,
                                                 bool keepDims,
                                                 Func<LLIRValue, LLIRValue, LLIRValue> reductionOp,
                                                 LLIRValue identityValue)
        {
            var bodyBlock = new IRBlock("reduce_body");

            // Initialize result to identity
            var resultVal = new RegisterValue(result.Type, "result");
            bodyBlock.AddOperation(new AddScalarOp(resultVal, identityValue, resultVal));

            // Loop over reduction axes
            foreach (int axis in axes)
            {
                var iv = new RegisterValue(new ScalarType(DataType.Int32), $"axis_{axis}");
                // Generate loop for this axis
                // Update accumulator in loop body
            }

            // Store result
            bodyBlock.AddOperation(new StoreOp(result, resultVal));

            // Placeholder return - full implementation would generate actual reduction loop structure
            return null;
        }

        #endregion

        #region Address Computation

        /// <summary>
        /// Computes the memory address for a given base pointer and index.
        /// </summary>
        private LLIRValue ComputeAddress(MemoryValue basePtr, LLIRValue index)
        {
            // address = basePtr + index * elementSize (simplified)
            var indexOffset = new RegisterValue(new ScalarType(DataType.Int32), "offset");

            // offset = index * elementSize (simplified - assume 4 bytes)
            var elementSizeConst = new RegisterValue(new ScalarType(DataType.Int32), "4");
            var mulOp = new MulScalarOp(index, elementSizeConst, indexOffset);

            // address = basePtr + offset
            var address = new RegisterValue(new PointerType(basePtr.Type), "addr");
            var addOp = new AddScalarOp(basePtr, indexOffset, address);

            return address;
        }

        /// <summary>
        /// Computes the N-dimensional memory address for a given base pointer and indices.
        /// </summary>
        private LLIRValue ComputeNDAddress(MemoryValue basePtr, LLIRValue[] indices,
                                           int[] strides, int elementSize)
        {
            // address = basePtr + (indices[0] * strides[0] + indices[1] * strides[1] + ...) * elementSize
            // Simplified for row-major layout

            var offset = Constant(0);
            var currentOffset = offset;

            for (int i = 0; i < indices.Length; i++)
            {
                var strideConst = Constant(strides[i]);
                var term = new RegisterValue(new ScalarType(DataType.Int32), $"term_{i}");
                var mulOp = new MulScalarOp(indices[i], strideConst, term);
                var accAddOp = new AddScalarOp(currentOffset, term, currentOffset);
            }

            var totalOffset = new RegisterValue(new ScalarType(DataType.Int32), "total_offset");
            var scaledOffset = new RegisterValue(new ScalarType(DataType.Int32), "scaled_offset");
            var scaleMulOp = new MulScalarOp(currentOffset, Constant(elementSize), scaledOffset);

            var address = new RegisterValue(new PointerType(basePtr.Type), "addr");
            var addrAddOp = new AddScalarOp(basePtr, scaledOffset, address);

            return address;
        }

        #endregion

        #region Loop Lowering

        /// <summary>
        /// Lowers a ForLoopOp to LLIR.
        /// </summary>
        private IROperation LowerForLoopOp(IRContext targetContext, ForLoopOp loopOp)
        {
            var rewriter = new OperationRewriter(loopOp.Context, targetContext);

            // Lower MLIR ForLoopOp to LLIR LLIRForLoopOp
            // Remap induction variable and body

            var start = rewriter.RemapValue(loopOp.LowerBound) as LLIRValue;
            var end = rewriter.RemapValue(loopOp.UpperBound) as LLIRValue;
            var step = rewriter.RemapValue(loopOp.Step) as LLIRValue;
            var iv = new RegisterValue(new ScalarType(DataType.Int32), loopOp.InductionVariable.Name);

            // Lower body block
            var loweredBody = rewriter.RemapBlock(loopOp.Body, targetContext);

            return new LLIRForLoopOp(start, end, step, iv, loweredBody, LoopUnrollHint.None);
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Creates a constant value in * given context.
        /// </summary>
        private RegisterValue Constant(int value)
        {
            var valueType = new ScalarType(DataType.Int32);
            return new RegisterValue(valueType, $"{value}");
        }

        /// <summary>
        /// Creates a constant float value in * given context.
        /// </summary>
        private RegisterValue Constant(float value)
        {
            var valueType = new ScalarType(DataType.Float32);
            return new RegisterValue(valueType, $"{value}f");
        }

        /// <summary>
        /// Creates a constant float value in the given context.
        /// </summary>
        private RegisterValue Constant(IRContext context, float value)
        {
            var valueType = new ScalarType(DataType.Float32);
            return new RegisterValue(valueType, $"{value}f");
        }

        #endregion
    }
}
