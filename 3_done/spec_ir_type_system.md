# Spec: IR Type System and Core Abstractions

## Overview
Define the foundational type system and abstract base classes for the entire IR infrastructure. This provides the common interfaces and data structures used across all IR levels (HLIR, MLIR, LLIR, Backend IRs).

## Requirements

### Core Type Classes

**IIRType Interface**
```csharp
public interface IIRType
{
    string Name { get; }
    bool Equals(IIRType other);
    int GetHashCode();
    IIRType Canonicalize();
}
```

**TensorType**
```csharp
public class TensorType : IIRType
{
    public DataType ElementType { get; }
    public int[] Shape { get; }  // Can contain -1 for dynamic dimensions
    public bool IsDynamic { get; }
    public int Rank { get; }

    public TensorType(DataType elementType, int[] shape);
    public TensorType WithNewShape(int[] newShape);
    public bool HasKnownShape();
}
```

**DataType Enum**
```csharp
public enum DataType
{
    Float32, Float64, Float16, BFloat16,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Bool
}
```

### IR Value and Attribute Abstractions

**IRValue**
```csharp
public class IRValue
{
    public IIRType Type { get; }
    public string Name { get; }
    public int Id { get; }

    protected IRValue(IIRType type, string name);
}
```

**IRAttribute**
```csharp
public interface IIRAttribute
{
    IIRType Type { get; }
    object Value { get; }
}

public class TensorAttribute : IIRAttribute { ... }
public class FloatAttribute : IIRAttribute { ... }
public class IntAttribute : IIRAttribute { ... }
public class BoolAttribute : IIRAttribute { ... }
public class ArrayAttribute : IIRAttribute { ... }
```

### IR Operation Base Classes

**IROperation**
```csharp
public abstract class IROperation
{
    public string Name { get; }
    public IRValue[] Operands { get; }
    public IRValue[] Results { get; }
    public IROpcode Opcode { get; }
    public IIRType[] ResultTypes { get; }

    protected IROperation(string name, IROpcode opcode,
                          IRValue[] operands, IIRType[] resultTypes);

    public abstract void Validate();
    public abstract IROperation Clone();
}
```

**IROpcode**
```csharp
public enum IROpcode
{
    // High-level ops
    MatMul, Conv2D, Add, Sub, Mul, Div,
    // Control flow
    IfOp, LoopOp, ScanOp,
    // Memory
    Alloc, Load, Store,
    // Reduction
    ReduceSum, ReduceMean, ReduceMax,
    // Backend-specific ranges defined separately
}
```

### IR Context

**IRContext**
```csharp
public class IRContext
{
    private int _nextValueId = 0;
    private Dictionary<int, IRValue> _values;
    private Dictionary<int, IROperation> _operations;

    public IRValue CreateValue(IIRType type, string name = null);
    public void RegisterOperation(IROperation op);
    public IRValue GetValue(int id);
    public IROperation GetOperation(int id);
}
```

## Implementation Details

1. **Type Unification**: Implement type compatibility checking methods in `TensorType`
2. **Shape Operations**: Helper methods for shape inference and manipulation
3. **Type Caching**: Use static caches for common tensor types to reduce allocations
4. **Validation**: Each `IROperation` must validate operand count and types

## Deliverables

- `src/IR/Types/IIRType.cs`
- `src/IR/Types/TensorType.cs`
- `src/IR/Types/DataType.cs`
- `src/IR/Values/IRValue.cs`
- `src/IR/Attributes/IIRAttribute.cs`
- `src/IR/Attributes/ConcreteAttributes.cs`
- `src/IR/Operations/IROperation.cs`
- `src/IR/Operations/IROpcode.cs`
- `src/IR/IRContext.cs`

## Success Criteria

- All type classes compile and pass basic validation
- Tensor types correctly handle dynamic shapes
- IRContext can create and track values and operations
- Attribute system supports all required scalar and tensor attributes

## Dependencies

- None (foundational)
