# Spec: High-Level IR (HLIR) Operation Set

## Overview
Define the concrete set of operations for the High-Level IR (HLIR) that captures model semantics in a hardware-agnostic form. These operations map directly to common tensor operations and control flow structures used in ML models.

## Requirements

### Tensor Operations

**Elementwise Operations**
```csharp
public class AddOp : IROperation
{
    public AddOp(IRValue lhs, IRValue rhs, IRValue result);
    public static IRValue Create(IRContext ctx, IRValue lhs, IRValue rhs);
}

public class SubOp : IROperation { /* Similar structure */ }
public class MulOp : IROperation { /* Similar structure */ }
public class DivOp : IROperation { /* Similar structure */ }

// Activation functions
public class ReLUOp : IROperation
{
    public ReLUOp(IRValue input, IRValue result);
}

public class SigmoidOp : IROperation { /* Similar structure */ }
public class TanhOp : IROperation { /* Similar structure */ }
public class GELUOp : IROperation { /* Similar structure */ }
```

**Matrix Operations**
```csharp
public class MatMulOp : IROperation
{
    public bool TransposeA { get; }
    public bool TransposeB { get; }

    public MatMulOp(IRValue lhs, IRValue rhs, IRValue result,
                    bool transposeA = false, bool transposeB = false);
}

public class Conv2DOp : IROperation
{
    public int[] KernelSize { get; }
    public int[] Stride { get; }
    public int[] Padding { get; }
    public int[] Dilation { get; }
    public int Groups { get; }

    public Conv2DOp(IRValue input, IRValue weight, IRValue bias,
                    IRValue result, int[] kernelSize, int[] stride,
                    int[] padding = null, int[] dilation = null,
                    int groups = 1);
}
```

**Pooling Operations**
```csharp
public class MaxPool2DOp : IROperation
{
    public int[] KernelSize { get; }
    public int[] Stride { get; }
    public int[] Padding { get; }

    public MaxPool2DOp(IRValue input, IRValue result,
                       int[] kernelSize, int[] stride, int[] padding = null);
}

public class AvgPool2DOp : IROperation { /* Similar structure */ }
```

**Reduction Operations**
```csharp
public class ReduceSumOp : IROperation
{
    public int[] Axes { get; }
    public bool KeepDims { get; }

    public ReduceSumOp(IRValue input, IRValue result,
                      int[] axes, bool keepDims = false);
}

public class ReduceMeanOp : IROperation { /* Similar structure */ }
public class ReduceMaxOp : IROperation { /* Similar structure */ }
```

### Shape Operations

```csharp
public class ReshapeOp : IROperation
{
    public int[] NewShape { get; }

    public ReshapeOp(IRValue input, IRValue result, int[] newShape);
}

public class TransposeOp : IROperation
{
    public int[] Permutation { get; }

    public TransposeOp(IRValue input, IRValue result, int[] permutation);
}

public class BroadcastToOp : IROperation
{
    public int[] TargetShape { get; }

    public BroadcastToOp(IRValue input, IRValue result, int[] targetShape);
}
```

### Control Flow Operations

```csharp
public class IfOp : IROperation
{
    public IRValue Condition { get; }
    public IRBlock TrueBlock { get; }
    public IRBlock FalseBlock { get; }

    public IfOp(IRValue condition, IRValue result,
                IRBlock trueBlock, IRBlock falseBlock);
}

public class LoopOp : IROperation
{
    public IRValue InitialValue { get; }
    public IRValue LoopCondition { get; }
    public IRBlock Body { get; }

    public LoopOp(IRValue initialValue, IRValue result,
                  IRValue loopCondition, IRBlock body);
}
```

### Constant Operation

```csharp
public class ConstantOp : IROperation
{
    public IIRAttribute Value { get; }
    public IRValue Output { get; }

    public ConstantOp(IRValue output, IIRAttribute value);
}
```

## Implementation Details

1. **Factory Methods**: Each operation class should have static `Create` methods for convenient instantiation
2. **Type Validation**: Validate that operand types are compatible (e.g., MatMul requires compatible matrix shapes)
3. **Shape Inference**: Operations should be able to infer output shapes where possible
4. **Operation Registry**: Auto-register operations with the IR context

## Deliverables

- `src/IR/HLIR/Elementwise/AddOp.cs`
- `src/IR/HLIR/Elementwise/SubOp.cs`
- `src/IR/HLIR/Elementwise/MulOp.cs`
- `src/IR/HLIR/Elementwise/DivOp.cs`
- `src/IR/HLIR/Activation/ReLUOp.cs`
- `src/IR/HLIR/Activation/SigmoidOp.cs`
- `src/IR/HLIR/Activation/TanhOp.cs`
- `src/IR/HLIR/Matrix/MatMulOp.cs`
- `src/IR/HLIR/Conv/Conv2DOp.cs`
- `src/IR/HLIR/Pool/MaxPool2DOp.cs`
- `src/IR/HLIR/Pool/AvgPool2DOp.cs`
- `src/IR/HLIR/Reduce/ReduceSumOp.cs`
- `src/IR/HLIR/Reduce/ReduceMeanOp.cs`
- `src/IR/HLIR/Reduce/ReduceMaxOp.cs`
- `src/IR/HLIR/Shape/ReshapeOp.cs`
- `src/IR/HLIR/Shape/TransposeOp.cs`
- `src/IR/HLIR/Shape/BroadcastToOp.cs`
- `src/IR/HLIR/ControlFlow/IfOp.cs`
- `src/IR/HLIR/ControlFlow/LoopOp.cs`
- `src/IR/HLIR/ConstantOp.cs`

## Success Criteria

- All operations correctly validate operand types
- Operations can be created and added to IRContext
- Factory methods work correctly
- Basic shape inference works for operations where applicable

## Dependencies

- spec_ir_type_system.md
