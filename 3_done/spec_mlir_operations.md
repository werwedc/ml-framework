# Spec: Mid-Level IR (MLIR) Operation Set

## Overview
Define the Mid-Level IR (MLIR) with normalized operations. MLIR provides a lowered representation with standardized operation signatures, enabling more aggressive optimizations like loop transformations and memory planning.

## Requirements

### Normalized Operation Base

```csharp
public class MLIR : IROperation
{
    // All MLIR operations inherit from IROperation but follow strict normalization rules
    // - Single output tensor
    // - Consistent shape inference
    // - No fused operations (e.g., separate Conv+BN)
}
```

### Normalized Tensor Operations

**Elementwise Operations (Broadcast-aware)**
```csharp
public class BroadcastAddOp : IROperation
{
    public IRValue Lhs { get; }
    public IRValue Rhs { get; }
    public IRValue Result { get; }
    public int[] BroadcastShape { get; }

    // Result shape is explicitly specified, not inferred
    public BroadcastAddOp(IRValue lhs, IRValue rhs, IRValue result, int[] broadcastShape);
}
```

**Normalized Convolution**
```csharp
public class ConvOp : IROperation
{
    public IRValue Input { get; }
    public IRValue Weight { get; }
    public IRValue Result { get; }

    // Explicit specification of all parameters
    public int[] InputShape { get; }
    public int[] WeightShape { get; }
    public int[] OutputShape { get; }
    public int[] KernelSize { get; }
    public int[] Stride { get; }
    public int[] Padding { get; }
    public int[] Dilation { get; }
    public int Groups { get; }

    public ConvOp(IRValue input, IRValue weight, IRValue result,
                 int[] inputShape, int[] weightShape, int[] outputShape,
                 int[] kernelSize, int[] stride, int[] padding,
                 int[] dilation, int groups);
}
```

**Explicit Memory Operations**
```csharp
public class AllocTensorOp : IROperation
{
    public IRValue Result { get; }
    public TensorType AllocatedType { get; }

    public AllocTensorOp(IRValue result, TensorType allocatedType);
}

public class DeallocTensorOp : IROperation
{
    public IRValue Tensor { get; }

    public DeallocTensorOp(IRValue tensor);
}
```

### Explicit Loop Operations

```csharp
public class ForLoopOp : IROperation
{
    public IRValue LowerBound { get; }
    public IRValue UpperBound { get; }
    public IRValue Step { get; }
    public IRValue InductionVariable { get; }
    public IRBlock Body { get; }

    public ForLoopOp(IRValue lowerBound, IRValue upperBound, IRValue step,
                     IRValue inductionVariable, IRBlock body);
}

public class ParallelForLoopOp : IROperation
{
    // Similar to ForLoopOp but marked for parallel execution
    public int NumThreads { get; }

    public ParallelForLoopOp(IRValue lowerBound, IRValue upperBound, IRValue step,
                            IRValue inductionVariable, IRBlock body, int numThreads);
}
```

### Reduction Operations (Normalized)

```csharp
public class ReduceOp : IROperation
{
    public IRValue Input { get; }
    public IRValue Result { get; }
    public ReductionKind Kind { get; }
    public int[] Axes { get; }
    public bool KeepDims { get; }

    public ReduceOp(IRValue input, IRValue result, ReductionKind kind,
                   int[] axes, bool keepDims);
}

public enum ReductionKind
{
    Sum, Mean, Max, Min, Prod, Any, All
}
```

### Index and Slice Operations

```csharp
public class SliceOp : IROperation
{
    public IRValue Input { get; }
    public IRValue Result { get; }
    public int[] Starts { get; }
    public int[] Ends { get; }
    public int[] Strides { get; }

    public SliceOp(IRValue input, IRValue result,
                  int[] starts, int[] ends, int[] strides);
}

public class GatherOp : IROperation
{
    public IRValue Input { get; }
    public IRValue Indices { get; }
    public IRValue Result { get; }
    public int Axis { get; }

    public GatherOp(IRValue input, IRValue indices, IRValue result, int axis);
}

public class ScatterOp : IROperation
{
    public IRValue Input { get; }
    public IRValue Updates { get; }
    public IRValue Indices { get; }
    public IRValue Result { get; }
    public int Axis { get; }

    public ScatterOp(IRValue input, IRValue updates, IRValue indices,
                    IRValue result, int axis);
}
```

### Type Conversion Operations

```csharp
public class CastOp : IROperation
{
    public IRValue Input { get; }
    public IRValue Result { get; }
    public DataType TargetType { get; }

    public CastOp(IRValue input, IRValue result, DataType targetType);
}
```

### Shape Operations (Explicit)

```csharp
public class DynamicReshapeOp : IROperation
{
    public IRValue Input { get; }
    public IRValue Shape { get; }  // Shape tensor, not static array
    public IRValue Result { get; }

    public DynamicReshapeOp(IRValue input, IRValue shape, IRValue result);
}
```

## Implementation Details

1. **Normalization Rules**:
   - All operations have explicit output shapes
   - No implicit broadcasting (use BroadcastAddOp instead of AddOp)
   - Fused operations are decomposed
   - Control flow is explicit (no if-else expressions, only IfOp)

2. **Shape Verification**: All operations validate input/output shapes match

3. **Metadata**: MLIR operations can carry additional metadata for optimizations

## Deliverables

- `src/IR/MLIR/Elementwise/BroadcastAddOp.cs`
- `src/IR/MLIR/Conv/ConvOp.cs`
- `src/IR/MLIR/Memory/AllocTensorOp.cs`
- `src/IR/MLIR/Memory/DeallocTensorOp.cs`
- `src/IR/MLIR/Loop/ForLoopOp.cs`
- `src/IR/MLIR/Loop/ParallelForLoopOp.cs`
- `src/IR/MLIR/Reduce/ReduceOp.cs`
- `src/IR/MLIR/Index/SliceOp.cs`
- `src/IR/MLIR/Index/GatherOp.cs`
- `src/IR/MLIR/Index/ScatterOp.cs`
- `src/IR/MLIR/Type/CastOp.cs`
- `src/IR/MLIR/Shape/DynamicReshapeOp.cs`

## Success Criteria

- All MLIR operations enforce normalization rules
- Explicit memory operations track allocations
- Loop operations capture control flow explicitly
- All operations validate shapes correctly

## Dependencies

- spec_ir_type_system.md
- spec_hlir_operations.md
