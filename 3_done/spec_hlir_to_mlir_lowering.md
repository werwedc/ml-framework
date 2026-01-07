# Spec: HLIR to MLIR Lowering Pass

## Overview
Implement the lowering pass that transforms High-Level IR to Mid-Level IR. This pass normalizes operations, decomposes fused operations, and makes shapes and broadcasting explicit.

## Requirements

### HLIRtoMLIRLoweringPass

```csharp
public class HLIRtoMLIRLoweringPass : LoweringPassBase
{
    public HLIRtoMLIRLoweringPass()
        : base("HLIR", "MLIR") { }

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

    public override IROperation Lower(IRContext targetContext, IROperation op)
    {
        // Dispatch to specific lowering methods
        switch (op)
        {
            case AddOp addOp: return LowerAddOp(targetContext, addOp);
            case SubOp subOp: return LowerSubOp(targetContext, subOp);
            case MatMulOp matMulOp: return LowerMatMulOp(targetContext, matMulOp);
            case Conv2DOp convOp: return LowerConv2DOp(targetContext, convOp);
            // ... other cases
            default: throw new NotSupportedException($"Cannot lower {op.Name}");
        }
    }
}
```

### Lowering Strategy: Elementwise Operations

```csharp
private IROperation LowerAddOp(IRContext targetContext, AddOp addOp)
{
    var rewriter = new OperationRewriter(addOp.Context, targetContext);

    // 1. Infer broadcast shape if needed
    var lhsType = addOp.Lhs.Type as TensorType;
    var rhsType = addOp.Rhs.Type as TensorType;
    int[] broadcastShape = InferBroadcastShape(lhsType.Shape, rhsType.Shape);

    // 2. If shapes differ, insert BroadcastToOp
    if (!ShapesEqual(lhsType.Shape, broadcastShape))
    {
        var broadcastLhs = CreateBroadcastOp(targetContext, addOp.Lhs, broadcastShape);
        rewriter.SetMapping(addOp.Lhs, broadcastLhs.Result);
    }

    if (!ShapesEqual(rhsType.Shape, broadcastShape))
    {
        var broadcastRhs = CreateBroadcastOp(targetContext, addOp.Rhs, broadcastShape);
        rewriter.SetMapping(addOp.Rhs, broadcastRhs.Result);
    }

    // 3. Create BroadcastAddOp
    var resultType = new TensorType(lhsType.ElementType, broadcastShape);
    var result = targetContext.CreateValue(resultType);
    return new BroadcastAddOp(
        rewriter.RemapValue(addOp.Lhs),
        rewriter.RemapValue(addOp.Rhs),
        result,
        broadcastShape
    );
}
```

### Lowering Strategy: Convolution

```csharp
private IROperation LowerConv2DOp(IRContext targetContext, Conv2DOp convOp)
{
    var rewriter = new OperationRewriter(convOp.Context, targetContext);

    var inputType = convOp.Input.Type as TensorType;
    var weightType = convOp.Weight.Type as TensorType;

    // 1. Explicitly compute output shape
    int[] outputShape = ComputeConvOutputShape(
        inputType.Shape,
        weightType.Shape,
        convOp.Stride,
        convOp.Padding,
        convOp.Dilation
    );

    // 2. Insert explicit padding if needed
    var paddedInput = convOp.Input;
    if (!PaddingIsZero(convOp.Padding))
    {
        paddedInput = CreateExplicitPadding(targetContext, convOp.Input, convOp.Padding);
    }

    // 3. Create normalized ConvOp
    var resultType = new TensorType(inputType.ElementType, outputShape);
    var result = targetContext.CreateValue(resultType);
    return new ConvOp(
        rewriter.RemapValue(paddedInput),
        rewriter.RemapValue(convOp.Weight),
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
}
```

### Lowering Strategy: Fused Operations

```csharp
private IROperation LowerConv2DOpWithBias(IRContext targetContext, Conv2DOp convOp)
{
    // Conv2DOp with bias is a fused operation
    // Lower to ConvOp + AddOp (bias addition)

    var convResult = LowerConv2DOpWithoutBias(targetContext, convOp);

    // Add bias as a separate operation
    var biasBroadcastShape = new int[] { convResult.Result.Type.Shape[0], 1, 1 };
    var broadcastBias = CreateBroadcastOp(targetContext, convOp.Bias, biasBroadcastShape);

    var resultType = convResult.Result.Type;
    var result = targetContext.CreateValue(resultType);
    return new BroadcastAddOp(
        convResult.Result,
        broadcastBias.Result,
        result,
        resultType.Shape
    );
}
```

### Lowering Strategy: Pooling

```csharp
private IROperation LowerMaxPool2DOp(IRContext targetContext, MaxPool2DOp poolOp)
{
    // Lower MaxPool2D to ReduceMax over sliding windows
    // This involves creating explicit loops

    var inputType = poolOp.Input.Type as TensorType;
    int[] outputShape = ComputePoolOutputShape(inputType.Shape, poolOp.KernelSize, poolOp.Stride, poolOp.Padding);

    var resultType = new TensorType(inputType.ElementType, outputShape);
    var result = targetContext.CreateValue(resultType);

    // Create a for loop for the sliding window
    var iv = targetContext.CreateValue(new TensorType(DataType.Int32, new[] { 1 }));
    var body = new IRBlock("pool_body");

    // ... implement sliding window logic using explicit loops and ReduceOp

    var loop = new ForLoopOp(
        Constant(targetContext, 0),
        Constant(targetContext, outputShape[1] * outputShape[2]),
        Constant(targetContext, 1),
        iv,
        body
    );

    return loop;
}
```

### Lowering Strategy: Activation Functions

```csharp
private IROperation LowerReLUOp(IRContext targetContext, ReLUOp reluOp)
{
    // Lower ReLU(x) to max(x, 0)
    var rewriter = new OperationRewriter(reluOp.Context, targetContext);

    var zero = Constant(targetContext, 0.0f);
    var resultType = reluOp.Result.Type;
    var result = targetContext.CreateValue(resultType);

    // Create explicit max operation (if not already available)
    return CreateMaxOp(
        rewriter.RemapValue(reluOp.Input),
        zero,
        result
    );
}
```

### Shape Inference Helpers

```csharp
private int[] InferBroadcastShape(int[] shape1, int[] shape2)
{
    // Implement numpy-style broadcasting rules
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
            throw new InvalidOperationException($"Incompatible shapes: {string.Join(",", shape1)} and {string.Join(",", shape2)}");
        }
    }

    return result;
}

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
```

## Implementation Details

1. **Value Mapping**: Use `OperationRewriter` to track mappings between HLIR and MLIR values
2. **Type Preservation**: Preserve tensor data types during lowering
3. **Error Handling**: Provide clear error messages for unsupported operations
4. **Validation**: Verify that lowered operations are valid MLIR

## Deliverables

- `src/IR/Lowering/HLIRtoMLIRLoweringPass.cs`

## Success Criteria

- All HLIR operations successfully lower to MLIR
- Broadcasting is made explicit
- Fused operations are decomposed
- Output shapes are correctly inferred and explicit

## Dependencies

- spec_ir_type_system.md
- spec_hlir_operations.md
- spec_mlir_operations.md
- spec_ir_transformation_infra.md
