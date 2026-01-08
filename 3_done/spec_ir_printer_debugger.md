# Spec: IR Printer and Debugger

## Overview
Implement printing and debugging utilities for IR inspection at any stage of the compilation pipeline. This enables developers to visualize IR transformations and debug optimization passes.

## Requirements

### IRPrinter Interface

```csharp
public interface IIRPrinter
{
    string Print(HLIRModule module);
    string Print(HLIRFunction function);
    string Print(IRBlock block);
    string Print(IROperation op);
}
```

### Text-Based IR Printer

```csharp
public class TextIRPrinter : IIRPrinter
{
    private StringBuilder _sb;
    private int _indentLevel;
    private bool _printAttributes;
    private bool _printTypes;

    public TextIRPrinter(bool printAttributes = true, bool printTypes = true)
    {
        _printAttributes = printAttributes;
        _printTypes = printTypes;
    }

    public string Print(HLIRModule module)
    {
        _sb = new StringBuilder();
        _indentLevel = 0;

        AppendLine($"module '{module.Context.GetHashCode()}' {{");
        Indent();

        // Print constants
        foreach (var (name, value) in module.Constants)
        {
            AppendLine($"constant {name} = {PrintAttribute(value)}");
        }

        // Print functions
        foreach (var function in module.Functions)
        {
            Print(function);
        }

        Dedent();
        AppendLine("}");

        return _sb.ToString();
    }

    public string Print(HLIRFunction function)
    {
        AppendLine($"function {function.Name}(");
        Indent();

        // Print parameters
        for (int i = 0; i < function.Parameters.Count; i++)
        {
            var param = function.Parameters[i];
            AppendLine($"  %{param.Id}: {PrintType(param.Type)}{GetComma(i, function.Parameters.Count)}");
        }

        Dedent();
        AppendLine(") -> {");
        Indent();

        // Print body blocks
        Print(function.Body);

        Dedent();
        AppendLine("}");

        return _sb.ToString();
    }

    public string Print(IRBlock block)
    {
        AppendLine($"block {block.Name}:");
        Indent();

        // Print arguments
        foreach (var arg in block.Arguments)
        {
            AppendLine($"  %{arg.Id}: {PrintType(arg.Type)} // argument");
        }

        // Print operations
        foreach (var op in block.Operations)
        {
            Print(op);
        }

        // Print returns
        if (block.Returns.Count > 0)
        {
            Append($"  return");
            foreach (var ret in block.Returns)
            {
                Append($" %{ret.Id}");
            }
            AppendLine();
        }

        Dedent();
        return _sb.ToString();
    }

    public string Print(IROperation op)
    {
        Append($"  %{op.Results[0].Id} = ");
        Append($"{op.Name}");

        // Print operands
        if (op.Operands.Length > 0)
        {
            Append("(");
            for (int i = 0; i < op.Operands.Length; i++)
            {
                Append($"%{op.Operands[i].Id}");
                if (_printTypes && op.Operands[i].Type != null)
                {
                    Append($": {PrintType(op.Operands[i].Type)}");
                }
                if (i < op.Operands.Length - 1)
                    Append(", ");
            }
            Append(")");
        }

        // Print operation-specific attributes
        PrintOperationAttributes(op);

        AppendLine();

        return _sb.ToString();
    }

    private void PrintOperationAttributes(IROperation op)
    {
        switch (op)
        {
            case Conv2DOp conv:
                Append($" kernel={PrintArray(conv.KernelSize)}");
                Append($" stride={PrintArray(conv.Stride)}");
                Append($" padding={PrintArray(conv.Padding)}");
                break;
            case MatMulOp matmul:
                if (matmul.TransposeA) Append(" transpose_a=true");
                if (matmul.TransposeB) Append(" transpose_b=true");
                break;
            case ReshapeOp reshape:
                Append($" new_shape={PrintArray(reshape.NewShape)}");
                break;
            // ... other operations
        }
    }

    private string PrintType(IIRType type)
    {
        if (type is TensorType tensorType)
        {
            return $"{tensorType.ElementType}{PrintShape(tensorType.Shape)}";
        }
        return type.ToString();
    }

    private string PrintShape(int[] shape)
    {
        return $"[{string.Join(", ", shape)}]";
    }

    private string PrintArray<T>(T[] array)
    {
        return $"[{string.Join(", ", array)}]";
    }

    private string PrintAttribute(IIRAttribute attr)
    {
        // Format attribute value appropriately
        return attr.Value.ToString();
    }

    private void Indent() => _indentLevel++;
    private void Dedent() => _indentLevel--;

    private void Append(string str)
    {
        _sb.Append(str);
    }

    private void AppendLine(string str = "")
    {
        if (_indentLevel > 0)
        {
            _sb.Append(new string(' ', _indentLevel * 2));
        }
        _sb.AppendLine(str);
    }

    private string GetComma(int index, int count)
    {
        return index < count - 1 ? "," : "";
    }
}
```

### IR Debugger

```csharp
public class IRDebugger
{
    private HLIRModule _module;
    private Dictionary<int, IROperation> _operationsById;

    public IRDebugger(HLIRModule module)
    {
        _module = module;
        _operationsById = BuildOperationMap();
    }

    public IROperation FindOperation(int valueId)
    {
        // Find operation that produces given value
        foreach (var function in _module.Functions)
        {
            foreach (var block in EnumerateBlocks(function))
            {
                foreach (var op in block.Operations)
                {
                    foreach (var result in op.Results)
                    {
                        if (result.Id == valueId)
                            return op;
                    }
                }
            }
        }
        return null;
    }

    public List<IROperation> FindUses(int valueId)
    {
        var uses = new List<IROperation>();
        foreach (var function in _module.Functions)
        {
            foreach (var block in EnumerateBlocks(function))
            {
                foreach (var op in block.Operations)
                {
                    foreach (var operand in op.Operands)
                    {
                        if (operand.Id == valueId)
                        {
                            uses.Add(op);
                            break;
                        }
                    }
                }
            }
        }
        return uses;
    }

    public void PrintValueDefinition(int valueId)
    {
        var op = FindOperation(valueId);
        if (op != null)
        {
            var printer = new TextIRPrinter();
            Console.WriteLine($"Value %{valueId} is defined by:");
            Console.WriteLine(printer.Print(op));
        }
        else
        {
            Console.WriteLine($"Value %{valueId} is not defined (likely a function parameter)");
        }
    }

    public void PrintValueUses(int valueId)
    {
        var uses = FindUses(valueId);
        var printer = new TextIRPrinter();

        Console.WriteLine($"Value %{valueId} is used in {uses.Count} operation(s):");
        foreach (var use in uses)
        {
            Console.WriteLine(printer.Print(use));
        }
    }

    public void PrintFunctionStats(HLIRFunction function)
    {
        int opCount = 0;
        int valueCount = 0;
        var opTypes = new Dictionary<Type, int>();

        foreach (var block in EnumerateBlocks(function))
        {
            opCount += block.Operations.Count;
            foreach (var op in block.Operations)
            {
                opTypes.TryGetValue(op.GetType(), out int count);
                opTypes[op.GetType()] = count + 1;
                valueCount += op.Results.Length;
            }
        }

        Console.WriteLine($"Function: {function.Name}");
        Console.WriteLine($"  Operations: {opCount}");
        Console.WriteLine($"  Values: {valueCount}");
        Console.WriteLine($"  Blocks: {GetBlockCount(function)}");
        Console.WriteLine("  Operation types:");
        foreach (var (type, count) in opTypes.OrderByDescending(x => x.Value))
        {
            Console.WriteLine($"    {type.Name}: {count}");
        }
    }

    private Dictionary<int, IROperation> BuildOperationMap()
    {
        var map = new Dictionary<int, IROperation>();
        // Build map from value ID to operation
        return map;
    }

    private IEnumerable<IRBlock> EnumerateBlocks(HLIRFunction function)
    {
        yield return function.Body;
        // TODO: Enumerate nested blocks (e.g., in IfOp, LoopOp)
    }

    private int GetBlockCount(HLIRFunction function)
    {
        return EnumerateBlocks(function).Count();
    }
}
```

### IR Visualizer (Simple ASCII)

```csharp
public class ASCIIIRVisualizer
{
    public string VisualizeFunction(HLIRFunction function)
    {
        // Create simple ASCII visualization of function structure
        // Useful for quick debugging
        return "";
    }
}
```

## Deliverables

- `src/IR/Debug/IIRPrinter.cs`
- `src/IR/Debug/TextIRPrinter.cs`
- `src/IR/Debug/IRDebugger.cs`
- `src/IR/Debug/ASCIIIRVisualizer.cs`

## Success Criteria

- Can print entire module in readable format
- Can print individual functions, blocks, and operations
- Debugger can find operations and uses
- Output format is human-readable and well-indented

## Dependencies

- spec_ir_type_system.md
- spec_hlir_operations.md
- spec_hlir_graph_builder.md

## Example Output

```
module '12345' {
  constant weights = tensor<float32, [784, 128]>

  function mlp(%0: float32[32, 784], %1: float32[128, 10]) -> {
  block main:
    %0: float32[32, 784] // argument
    %1: float32[128, 10] // argument
    %2 = matmul(%0: float32[32, 784], %1: float32[784, 128])
    %3 = relu(%2: float32[32, 128])
    %4 = matmul(%3: float32[32, 128], %1: float32[128, 10])
    return %4
  }
}
```
