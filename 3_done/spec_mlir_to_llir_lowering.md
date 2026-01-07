# Spec: MLIR to LLIR Lowering Pass

## Overview
Implement the lowering pass that transforms Mid-Level IR to Low-Level IR. This pass makes memory operations explicit, introduces registers and buffers, and converts high-level operations to low-level arithmetic.

## Requirements

### MLIRtoLLIRLoweringPass

```csharp
public class MLIRtoLLIRLoweringPass : LoweringPassBase
{
    private MemoryLayout _memoryLayout;
    private int _vectorWidth;

    public MLIRtoLLIRLoweringPass(MemoryLayout memoryLayout = MemoryLayout.RowMajor,
                                  int vectorWidth = 1)
        : base("MLIR", "LLIR")
    {
        _memoryLayout = memoryLayout;
        _vectorWidth = vectorWidth;
    }

    public override bool CanLower(IROperation op)
    {
        // All MLIR operations should be lowerable to LLIR
        return op is BroadcastAddOp || op is ConvOp || op is AllocTensorOp ||
               op is ForLoopOp || op is ReduceOp || op is SliceOp ||
               op is CastOp || op is DynamicReshapeOp;
    }

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
}
```

### Lowering Strategy: Memory Allocation

```csharp
private IROperation LowerAllocTensorOp(IRContext targetContext, AllocTensorOp allocOp)
{
    var tensorType = allocOp.AllocatedType;
    int sizeInBytes = ComputeTensorSize(tensorType);
    int alignment = ComputeAlignment(tensorType.ElementType);

    var buffer = new MemoryValue(
        new PointerType(tensorType.ElementType),
        $"{allocOp.Result.Name}_buffer",
        0,  // Offset computed by memory allocator
        sizeInBytes
    );

    var allocOp = new AllocBufferOp(buffer, sizeInBytes, alignment);
    targetContext.RegisterOperation(allocOp);

    return allocOp;
}

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

private int ComputeAlignment(DataType elementType)
{
    // Align to element type size, minimum 16 bytes for SIMD
    return Math.Max(GetElementSize(elementType), 16);
}
```

### Lowering Strategy: Elementwise Operations

```csharp
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
            new RegisterValue(lhsType.ElementType, "lhs_val"),
            new RegisterValue(rhsType.ElementType, "rhs_val"),
            resultRegister
        )
    );
}

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

private IROperation Generate1DLoop(IRContext targetContext,
                                   MemoryValue lhs,
                                   MemoryValue rhs,
                                   MemoryValue result,
                                   int size,
                                   Func<LLIRValue, LLIRValue, LLIRValue, IROperation> bodyBuilder)
{
    var bodyBlock = new IRBlock("elemwise_body");
    var iv = new RegisterValue(new ScalarType(DataType.Int32), "i");
    var start = Constant(targetContext, 0);
    var end = Constant(targetContext, size);
    var step = Constant(targetContext, 1);

    // In loop body:
    // 1. Compute addresses: lhs + i * elementSize
    // 2. Load values
    // 3. Perform operation
    // 4. Store result

    var lhsAddr = ComputeAddress(lhs, iv);
    var rhsAddr = ComputeAddress(rhs, iv);
    var resultAddr = ComputeAddress(result, iv);

    var lhsVal = new RegisterValue(lhs.Type as ScalarType, "lhs_val");
    var rhsVal = new RegisterValue(rhs.Type as ScalarType, "rhs_val");
    var resultVal = new RegisterValue(result.Type as ScalarType, "result_val");

    bodyBlock.AddOperation(new LoadOp(lhsAddr, lhsVal));
    bodyBlock.AddOperation(new LoadOp(rhsAddr, rhsVal));
    bodyBlock.AddOperation(bodyBuilder(lhsVal, rhsVal, resultVal));
    bodyBlock.AddOperation(new StoreOp(resultAddr, resultVal));

    return new LLIRForLoopOp(start, end, step, iv, bodyBlock);
}
```

### Lowering Strategy: Convolution

```csharp
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
    bodyBlock.AddOperation(new AddScalarOp(accumulator, Constant(targetContext, 0.0f), accumulator));

    // Inner loops for kernel
    var kernelHeightIV = new RegisterValue(new ScalarType(DataType.Int32), "kh");
    var kernelWidthIV = new RegisterValue(new ScalarType(DataType.Int32), "kw");
    var inputChannelsIV = new RegisterValue(new ScalarType(DataType.Int32), "ic");

    // Generate complex nested loop structure
    // (simplified - full implementation would generate all nested loops)

    return bodyBlock;
}
```

### Lowering Strategy: Reduction Operations

```csharp
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
    var resultVal = new RegisterValue(result.Type as ScalarType, "result");
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

    return bodyBlock;
}
```

### Address Computation

```csharp
private LLIRValue ComputeAddress(MemoryValue basePtr, LLIRValue index, int elementSize)
{
    // address = basePtr + index * elementSize
    var indexOffset = new RegisterValue(new ScalarType(DataType.Int32), "offset");
    var elementSizeConst = Constant(index.Context, elementSize);

    // offset = index * elementSize
    var mulOp = new MulScalarOp(index, elementSizeConst, indexOffset);

    // address = basePtr + offset
    var address = new RegisterValue(new PointerType(basePtr.Type as PointerType), "addr");
    var addOp = new AddScalarOp(basePtr, indexOffset, address);

    return address;
}

private LLIRValue ComputeNDAddress(MemoryValue basePtr, LLIRValue[] indices,
                                   int[] strides, int elementSize)
{
    // address = basePtr + (indices[0] * strides[0] + indices[1] * strides[1] + ...) * elementSize
    // Simplified for row-major layout

    var offset = Constant(basePtr.Context, 0);
    var currentOffset = offset;

    for (int i = 0; i < indices.Length; i++)
    {
        var strideConst = Constant(indices[i].Context, strides[i]);
        var term = new RegisterValue(new ScalarType(DataType.Int32), $"term_{i}");
        var mulOp = new MulScalarOp(indices[i], strideConst, term);
        var addOp = new AddScalarOp(currentOffset, term, currentOffset);
    }

    var totalOffset = new RegisterValue(new ScalarType(DataType.Int32), "total_offset");
    var scaledOffset = new RegisterValue(new ScalarType(DataType.Int32), "scaled_offset");
    var scaleMulOp = new MulScalarOp(currentOffset, Constant(basePtr.Context, elementSize), scaledOffset);

    var address = new RegisterValue(new PointerType(basePtr.Type as PointerType), "addr");
    var addOp = new AddScalarOp(basePtr, scaledOffset, address);

    return address;
}
```

### Loop Lowering

```csharp
private IROperation LowerForLoopOp(IRContext targetContext, ForLoopOp loopOp)
{
    var rewriter = new OperationRewriter(loopOp.Context, targetContext);

    // Lower MLIR ForLoopOp to LLIR LLIRForLoopOp
    // Remap induction variable and body

    var start = rewriter.RemapValue(loopOp.LowerBound);
    var end = rewriter.RemapValue(loopOp.UpperBound);
    var step = rewriter.RemapValue(loopOp.Step);
    var iv = new RegisterValue(new ScalarType(DataType.Int32), loopOp.InductionVariable.Name);

    // Lower body block
    var loweredBody = rewriter.RemapBlock(loopOp.Body, targetContext);

    return new LLIRForLoopOp(start, end, step, iv, loweredBody, LoopUnrollHint.None);
}
```

## Implementation Details

1. **Memory Layout**: Support both row-major and column-major layouts
2. **Vectorization**: If vector width > 1, generate vector operations instead of scalar
3. **Register Allocation**: Simple register allocation for intermediate values
4. **Loop Unrolling**: Add hints for loop unrolling in appropriate cases

## Deliverables

- `src/IR/Lowering/MLIRtoLLIRLoweringPass.cs`

## Success Criteria

- All MLIR operations successfully lower to LLIR
- Memory operations are explicit
- Loops are correctly generated for tensor operations
- Address computation is correct for all tensor access patterns

## Dependencies

- spec_ir_type_system.md
- spec_mlir_operations.md
- spec_llir_foundation.md
- spec_ir_transformation_infra.md
