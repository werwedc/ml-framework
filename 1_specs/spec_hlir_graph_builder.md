# Spec: HLIR Graph Representation and Builder

## Overview
Implement the graph structure for High-Level IR and a builder API for constructing HLIR graphs from high-level model definitions. This provides the structural representation and convenient API for creating HLIR programs.

## Requirements

### IRBlock

```csharp
public class IRBlock
{
    public string Name { get; }
    public List<IROperation> Operations { get; }
    public List<IRValue> Arguments { get; }
    public List<IRValue> Returns { get; }

    public IRBlock(string name);
    public void AddOperation(IROperation op);
    public void AddArgument(IRValue arg);
    public void AddReturn(IRValue ret);
}
```

### HLIRFunction

```csharp
public class HLIRFunction
{
    public string Name { get; }
    public List<IRValue> Parameters { get; }
    public List<IRValue> Results { get; }
    public IRBlock Body { get; }
    public IRContext Context { get; }

    public HLIRFunction(string name, IRContext ctx);
    public IRValue AddParameter(TensorType type, string name);
    public void SetResults(params IRValue[] results);
}
```

### HLIRModule

```csharp
public class HLIRModule
{
    public IRContext Context { get; }
    public List<HLIRFunction> Functions { get; }
    public Dictionary<string, IIRAttribute> Constants { get; }

    public HLIRModule(IRContext ctx = null);
    public HLIRFunction CreateFunction(string name);
    public void AddConstant(string name, IIRAttribute value);
    public IIRAttribute GetConstant(string name);
}
```

### HLIRBuilder

```csharp
public class HLIRBuilder
{
    private IRContext _context;
    private HLIRFunction _currentFunction;
    private IRBlock _currentBlock;

    public HLIRBuilder(HLIRFunction function);
    public HLIRBuilder(HLIRModule module);

    // Function/block management
    public void SetInsertPoint(IRBlock block);
    public void SetInsertPointToEnd(IRBlock block);

    // Tensor operations
    public IRValue Add(IRValue lhs, IRValue rhs, string name = null);
    public IRValue Sub(IRValue lhs, IRValue rhs, string name = null);
    public IRValue Mul(IRValue lhs, IRValue rhs, string name = null);
    public IRValue Div(IRValue lhs, IRValue rhs, string name = null);

    // Activations
    public IRValue ReLU(IRValue input, string name = null);
    public IRValue Sigmoid(IRValue input, string name = null);

    // Matrix ops
    public IRValue MatMul(IRValue lhs, IRValue rhs,
                          bool transposeA = false, bool transposeB = false,
                          string name = null);

    public IRValue Conv2D(IRValue input, IRValue weight, IRValue bias,
                          int[] kernelSize, int[] stride,
                          int[] padding = null, int[] dilation = null,
                          int groups = 1, string name = null);

    // Pooling
    public IRValue MaxPool2D(IRValue input, int[] kernelSize,
                            int[] stride, int[] padding = null,
                            string name = null);

    // Reductions
    public IRValue ReduceSum(IRValue input, int[] axes,
                            bool keepDims = false, string name = null);

    // Shape ops
    public IRValue Reshape(IRValue input, int[] newShape, string name = null);
    public IRValue Transpose(IRValue input, int[] permutation, string name = null);
    public IRValue BroadcastTo(IRValue input, int[] targetShape, string name = null);

    // Constants
    public IRValue Constant(IIRAttribute value, string name = null);

    // Control flow
    public IRValue If(IRValue condition, Action<IRBlock> trueBranch,
                      Action<IRBlock> falseBranch, string name = null);

    public IRValue Loop(IRValue initialValue, Action<IRValue, IRBlock> body,
                        string name = null);
}
```

### Graph Traversal and Analysis

```csharp
public class HLIRGraphAnalyzer
{
    public static List<IROperation> TopologicalSort(HLIRFunction function);
    public static Dictionary<IRValue, List<IROperation>> FindUses(IRValue value);
    public static bool ValidateGraph(HLIRFunction function);
    public static List<IRValue> FindInputs(HLIRFunction function);
    public static List<IRValue> FindOutputs(HLIRFunction function);
}
```

## Implementation Details

1. **Block Naming**: Auto-generate unique names for blocks if not provided
2. **Value Naming**: Auto-generate names like "v0", "v1", etc. if not provided
3. **Graph Validation**: Ensure all values are defined before use
4. **Topological Sort**: Standard DFS-based algorithm for operation ordering
5. **Use-Def Chains**: Track which operations use which values

## Deliverables

- `src/IR/Graph/IRBlock.cs`
- `src/IR/Graph/HLIRFunction.cs`
- `src/IR/Graph/HLIRModule.cs`
- `src/IR/Graph/HLIRBuilder.cs`
- `src/IR/Graph/HLIRGraphAnalyzer.cs`

## Success Criteria

- Can build a simple linear graph using HLIRBuilder
- Graph validation catches undefined values
- Topological sort produces correct operation order
- Use-def chain analysis works correctly
- Functions can have multiple blocks for control flow

## Dependencies

- spec_ir_type_system.md
- spec_hlir_operations.md

## Example Usage

```csharp
var module = new HLIRModule();
var builder = new HLIRBuilder(module);

var func = module.CreateFunction("SimpleNN");
var input = func.AddParameter(new TensorType(DataType.Float32, new[] {32, 784}), "x");

var w1 = builder.Constant(new TensorAttribute(...));
var h1 = builder.ReLU(builder.MatMul(input, w1));

var w2 = builder.Constant(new TensorAttribute(...));
var output = builder.MatMul(h1, w2);

func.SetResults(output);
```
