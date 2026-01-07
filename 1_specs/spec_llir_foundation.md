# Spec: Low-Level IR (LLIR) Foundation

## Overview
Define the Low-Level IR (LLIR) that serves as the hardware-agnostic foundation for code generation. LLIR is close to LLVM IR, with explicit memory operations, buffer allocation, and low-level control flow.

## Requirements

### LLIR Value Types

```csharp
public class LLIRValue : IRValue
{
    public bool IsRegister { get; }
    public bool IsMemoryLocation { get; }

    public LLIRValue(IIRType type, string name, bool isRegister = false)
        : base(type, name)
    {
        IsRegister = isRegister;
        IsMemoryLocation = !isRegister;
    }
}

public class RegisterValue : LLIRValue
{
    public RegisterValue(IIRType type, string name) : base(type, name, isRegister: true) { }
}

public class MemoryValue : LLIRValue
{
    public int MemoryOffset { get; }
    public int SizeInBytes { get; }

    public MemoryValue(IIRType type, string name, int offset, int sizeInBytes)
        : base(type, name, isRegister: false)
    {
        MemoryOffset = offset;
        SizeInBytes = sizeInBytes;
    }
}
```

### LLIR Memory Operations

```csharp
public class AllocBufferOp : IROperation
{
    public LLIRValue Buffer { get; }
    public int SizeInBytes { get; }
    public int Alignment { get; }

    public AllocBufferOp(LLIRValue buffer, int sizeInBytes, int alignment = 16);
}

public class FreeBufferOp : IROperation
{
    public LLIRValue Buffer { get; }

    public FreeBufferOp(LLIRValue buffer);
}

public class LoadOp : IROperation
{
    public LLIRValue Address { get; }
    public LLIRValue Result { get; }
    public int Offset { get; }

    public LoadOp(LLIRValue address, LLIRValue result, int offset = 0);
}

public class StoreOp : IROperation
{
    public LLIRValue Address { get; }
    public LLIRValue Value { get; }
    public int Offset { get; }

    public StoreOp(LLIRValue address, LLIRValue value, int offset = 0);
}

public class MemcpyOp : IROperation
{
    public LLIRValue Dest { get; }
    public LLIRValue Src { get; }
    public int SizeInBytes { get; }

    public MemcpyOp(LLIRValue dest, LLIRValue src, int sizeInBytes);
}
```

### LLIR Arithmetic Operations (Scalar)

```csharp
public class AddScalarOp : IROperation
{
    public LLIRValue Lhs { get; }
    public LLIRValue Rhs { get; }
    public LLIRValue Result { get; }

    public AddScalarOp(LLIRValue lhs, LLIRValue rhs, LLIRValue result);
}

public class SubScalarOp : IROperation { /* Similar */ }
public class MulScalarOp : IROperation { /* Similar */ }
public class DivScalarOp : IROperation { /* Similar */ }
```

### LLIR Vector Operations

```csharp
public class VectorAddOp : IROperation
{
    public LLIRValue Lhs { get; }
    public LLIRValue Rhs { get; }
    public LLIRValue Result { get; }
    public int VectorWidth { get; }

    public VectorAddOp(LLIRValue lhs, LLIRValue rhs, LLIRValue result, int vectorWidth);
}

public class VectorMulOp : IROperation { /* Similar */ }
```

### LLIR Loop Operations

```csharp
public class LLIRForLoopOp : IROperation
{
    public LLIRValue Start { get; }
    public LLIRValue End { get; }
    public LLIRValue Step { get; }
    public LLIRValue InductionVariable { get; }
    public IRBlock Body { get; }
    public LoopUnrollHint UnrollHint { get; }

    public LLIRForLoopOp(LLIRValue start, LLIRValue end, LLIRValue step,
                        LLIRValue inductionVariable, IRBlock body,
                        LoopUnrollHint unrollHint = LoopUnrollHint.None);
}

public enum LoopUnrollHint
{
    None,
    Unroll,
    UnrollAndJam
}
```

### LLIR Control Flow

```csharp
public class BranchOp : IROperation
{
    public IRBlock Target { get; }

    public BranchOp(IRBlock target);
}

public class ConditionalBranchOp : IROperation
{
    public LLIRValue Condition { get; }
    public IRBlock TrueTarget { get; }
    public IRBlock FalseTarget { get; }

    public ConditionalBranchOp(LLIRValue condition, IRBlock trueTarget, IRBlock falseTarget);
}

public class ReturnOp : IROperation
{
    public LLIRValue ReturnValue { get; }  // Can be null

    public ReturnOp(LLIRValue returnValue = null);
}
```

### LLIR Phi Nodes (SSA Form)

```csharp
public class PhiNode : IROperation
{
    public LLIRValue Result { get; }
    public List<(IRBlock IncomingBlock, LLIRValue IncomingValue)> IncomingValues { get; }

    public PhiNode(LLIRValue result, List<(IRBlock, LLIRValue)> incomingValues);

    public void AddIncoming(IRBlock block, LLIRValue value);
}
```

### LLIR Type System

```csharp
public class ScalarType : IIRType
{
    public DataType DataType { get; }
    public bool IsFloat { get; }
    public bool IsInteger { get; }

    public ScalarType(DataType dataType);
}

public class PointerType : IIRType
{
    public IIRType ElementType { get; }

    public PointerType(IIRType elementType);
}

public class VectorType : IIRType
{
    public IIRType ElementType { get; }
    public int Width { get; }

    public VectorType(IIRType elementType, int width);
}
```

### LLIR Function

```csharp
public class LLIRFunction : HLIRFunction
{
    public bool IsKernel { get; }
    public List<LLIRValue> Registers { get; }
    public MemoryLayout MemoryLayout { get; }

    public LLIRFunction(string name, IRContext ctx, bool isKernel = false);

    public LLIRValue AllocateRegister(IIRType type, string name);
    public LLIRValue AllocateBuffer(int sizeInBytes, int alignment = 16);
}

public enum MemoryLayout
{
    RowMajor,
    ColumnMajor
}
```

## Implementation Details

1. **SSA Form**: LLIR should be in Static Single Assignment form
2. **Register Allocation**: Simple register allocation (can be improved later)
3. **Memory Layout**: Support both row-major and column-major layouts
4. **Type Safety**: Strict type checking at LLIR level

## Deliverables

- `src/IR/LLIR/Values/LLIRValue.cs`
- `src/IR/LLIR/Values/RegisterValue.cs`
- `src/IR/LLIR/Values/MemoryValue.cs`
- `src/IR/LLIR/Operations/Memory/AllocBufferOp.cs`
- `src/IR/LLIR/Operations/Memory/FreeBufferOp.cs`
- `src/IR/LLIR/Operations/Memory/LoadOp.cs`
- `src/IR/LLIR/Operations/Memory/StoreOp.cs`
- `src/IR/LLIR/Operations/Memory/MemcpyOp.cs`
- `src/IR/LLIR/Operations/Arithmetic/AddScalarOp.cs`
- `src/IR/LLIR/Operations/Arithmetic/SubScalarOp.cs`
- `src/IR/LLIR/Operations/Arithmetic/MulScalarOp.cs`
- `src/IR/LLIR/Operations/Arithmetic/DivScalarOp.cs`
- `src/IR/LLIR/Operations/Vector/VectorAddOp.cs`
- `src/IR/LLIR/Operations/Vector/VectorMulOp.cs`
- `src/IR/LLIR/Operations/ControlFlow/LLIRForLoopOp.cs`
- `src/IR/LLIR/Operations/ControlFlow/BranchOp.cs`
- `src/IR/LLIR/Operations/ControlFlow/ConditionalBranchOp.cs`
- `src/IR/LLIR/Operations/ControlFlow/ReturnOp.cs`
- `src/IR/LLIR/Operations/PhiNode.cs`
- `src/IR/LLIR/Types/ScalarType.cs`
- `src/IR/LLIR/Types/PointerType.cs`
- `src/IR/LLIR/Types/VectorType.cs`
- `src/IR/LLIR/LLIRFunction.cs`

## Success Criteria

- Can create LLIR functions with explicit memory operations
- Phi nodes correctly implement SSA form
- Register and memory value types work correctly
- Basic arithmetic and vector operations compile

## Dependencies

- spec_ir_type_system.md
- spec_hlir_graph_builder.md
