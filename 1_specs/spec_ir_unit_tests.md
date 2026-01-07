# Spec: IR System Unit Tests

## Overview
Implement comprehensive unit tests for the IR system. These tests ensure correctness of type system, operations, graph building, transformations, and lowering passes.

## Requirements

### Test Structure

```csharp
// tests/IR/Types/TensorTypeTests.cs
// tests/IR/Operations/HLIROperationTests.cs
// tests/IR/Operations/MLIROperationTests.cs
// tests/IR/Graph/GraphBuilderTests.cs
// tests/IR/Passes/OptimizationPassTests.cs
// tests/IR/Lowering/LoweringPassTests.cs
// tests/IR/Debug/IRPrinterTests.cs
```

### Type System Tests

**TensorTypeTests.cs**
```csharp
[TestFixture]
public class TensorTypeTests
{
    [Test]
    public void TensorType_Construction_CreatesCorrectType()
    {
        var type = new TensorType(DataType.Float32, new[] { 32, 784 });
        Assert.AreEqual(DataType.Float32, type.ElementType);
        Assert.AreEqual(new[] { 32, 784 }, type.Shape);
        Assert.AreEqual(2, type.Rank);
    }

    [Test]
    public void TensorType_WithDynamicShape_CreatesDynamicType()
    {
        var type = new TensorType(DataType.Float32, new[] { -1, 784 });
        Assert.IsTrue(type.IsDynamic);
        Assert.IsFalse(type.HasKnownShape());
    }

    [Test]
    public void TensorType_WithNewShape_CreatesNewType()
    {
        var original = new TensorType(DataType.Float32, new[] { 32, 784 });
        var reshaped = original.WithNewShape(new[] { 32 * 784 });
        Assert.AreEqual(new[] { 32 * 784 }, reshaped.Shape);
    }

    [Test]
    public void TensorType_Equals_ReturnsTrueForSameShape()
    {
        var type1 = new TensorType(DataType.Float32, new[] { 32, 784 });
        var type2 = new TensorType(DataType.Float32, new[] { 32, 784 });
        Assert.IsTrue(type1.Equals(type2));
    }

    [Test]
    public void TensorType_Equals_ReturnsFalseForDifferentShape()
    {
        var type1 = new TensorType(DataType.Float32, new[] { 32, 784 });
        var type2 = new TensorType(DataType.Float32, new[] { 64, 784 });
        Assert.IsFalse(type1.Equals(type2));
    }
}
```

### Operation Tests

**AddOpTests.cs**
```csharp
[TestFixture]
public class AddOpTests
{
    private IRContext _context;

    [SetUp]
    public void Setup()
    {
        _context = new IRContext();
    }

    [Test]
    public void AddOp_CreatesCorrectOperation()
    {
        var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
        var rhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
        var lhs = _context.CreateValue(lhsType, "lhs");
        var rhs = _context.CreateValue(rhsType, "rhs");
        var resultType = new TensorType(DataType.Float32, new[] { 32, 64 });
        var result = _context.CreateValue(resultType, "result");

        var addOp = new AddOp(lhs, rhs, result);

        Assert.AreEqual(lhs, addOp.Lhs);
        Assert.AreEqual(rhs, addOp.Rhs);
        Assert.AreEqual(result, addOp.Result);
    }

    [Test]
    public void AddOp_Validate_DoesNotThrowForValidShapes()
    {
        var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
        var rhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
        var lhs = _context.CreateValue(lhsType);
        var rhs = _context.CreateValue(rhsType);
        var result = _context.CreateValue(lhsType);

        var addOp = new AddOp(lhs, rhs, result);
        Assert.DoesNotThrow(() => addOp.Validate());
    }

    [Test]
    public void AddOp_Validate_ThrowsForIncompatibleShapes()
    {
        var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
        var rhsType = new TensorType(DataType.Float32, new[] { 32, 128 });
        var lhs = _context.CreateValue(lhsType);
        var rhs = _context.CreateValue(rhsType);
        var result = _context.CreateValue(lhsType);

        var addOp = new AddOp(lhs, rhs, result);
        Assert.Throws<InvalidOperationException>(() => addOp.Validate());
    }
}
```

**MatMulOpTests.cs**
```csharp
[TestFixture]
public class MatMulOpTests
{
    private IRContext _context;

    [SetUp]
    public void Setup()
    {
        _context = new IRContext();
    }

    [Test]
    public void MatMulOp_CreatesCorrectOperation()
    {
        var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
        var rhsType = new TensorType(DataType.Float32, new[] { 64, 128 });
        var resultType = new TensorType(DataType.Float32, new[] { 32, 128 });
        var lhs = _context.CreateValue(lhsType);
        var rhs = _context.CreateValue(rhsType);
        var result = _context.CreateValue(resultType);

        var matMulOp = new MatMulOp(lhs, rhs, result);

        Assert.AreEqual(lhs, matMulOp.Lhs);
        Assert.AreEqual(rhs, matMulOp.Rhs);
        Assert.AreEqual(result, matMulOp.Result);
        Assert.IsFalse(matMulOp.TransposeA);
        Assert.IsFalse(matMulOp.TransposeB);
    }

    [Test]
    public void MatMulOp_Validate_DoesNotThrowForValidShapes()
    {
        var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
        var rhsType = new TensorType(DataType.Float32, new[] { 64, 128 });
        var resultType = new TensorType(DataType.Float32, new[] { 32, 128 });
        var lhs = _context.CreateValue(lhsType);
        var rhs = _context.CreateValue(rhsType);
        var result = _context.CreateValue(resultType);

        var matMulOp = new MatMulOp(lhs, rhs, result);
        Assert.DoesNotThrow(() => matMulOp.Validate());
    }

    [Test]
    public void MatMulOp_Validate_ThrowsForIncompatibleInnerDimensions()
    {
        var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
        var rhsType = new TensorType(DataType.Float32, new[] { 128, 64 });
        var resultType = new TensorType(DataType.Float32, new[] { 32, 64 });
        var lhs = _context.CreateValue(lhsType);
        var rhs = _context.CreateValue(rhsType);
        var result = _context.CreateValue(resultType);

        var matMulOp = new MatMulOp(lhs, rhs, result);
        Assert.Throws<InvalidOperationException>(() => matMulOp.Validate());
    }
}
```

### Graph Building Tests

**GraphBuilderTests.cs**
```csharp
[TestFixture]
public class GraphBuilderTests
{
    private HLIRModule _module;
    private HLIRBuilder _builder;

    [SetUp]
    public void Setup()
    {
        _module = new HLIRModule();
        _builder = new HLIRBuilder(_module);
    }

    [Test]
    public void BuildSimpleLinearLayer_CreatesCorrectGraph()
    {
        var func = _module.CreateFunction("Linear");
        var input = func.AddParameter(new TensorType(DataType.Float32, new[] { 32, 784 }), "x");

        var w1 = _builder.Constant(new TensorAttribute(...), "w1");
        var h1 = _builder.ReLU(_builder.MatMul(input, w1));

        func.SetResults(h1);

        Assert.AreEqual(1, func.Parameters.Count);
        Assert.AreEqual(1, func.Results.Count);
        Assert.AreEqual(2, func.Body.Operations.Count);  // MatMul + ReLU
    }

    [Test]
    public void BuildMLP_CreatesCorrectGraph()
    {
        var func = _module.CreateFunction("MLP");
        var input = func.AddParameter(new TensorType(DataType.Float32, new[] { 32, 784 }), "x");

        var w1 = _builder.Constant(new TensorAttribute(...), "w1");
        var h1 = _builder.ReLU(_builder.MatMul(input, w1));

        var w2 = _builder.Constant(new TensorAttribute(...), "w2");
        var output = _builder.MatMul(h1, w2);

        func.SetResults(output);

        Assert.AreEqual(3, func.Body.Operations.Count);
    }

    [Test]
    public void BuildConvNet_CreatesCorrectGraph()
    {
        var func = _module.CreateFunction("ConvNet");
        var input = func.AddParameter(new TensorType(DataType.Float32, new[] { 1, 28, 28 }), "x");

        var conv1Weight = _builder.Constant(new TensorAttribute(...), "conv1_w");
        var conv1 = _builder.ReLU(_builder.Conv2D(input, conv1Weight, null,
                                                   new[] { 3, 3 }, new[] { 1, 1 },
                                                   new[] { 1, 1 }));

        var pool1 = _builder.MaxPool2D(conv1, new[] { 2, 2 }, new[] { 2, 2 });

        func.SetResults(pool1);

        Assert.AreEqual(3, func.Body.Operations.Count);
    }
}
```

### Optimization Pass Tests

**OptimizationPassTests.cs**
```csharp
[TestFixture]
public class OptimizationPassTests
{
    private HLIRModule _module;
    private HLIRBuilder _builder;

    [SetUp]
    public void Setup()
    {
        _module = new HLIRModule();
        _builder = new HLIRBuilder(_module);
    }

    [Test]
    public void ConstantFoldingPass_FoldsConstantAdd()
    {
        var func = _module.CreateFunction("Test");
        var const1 = _builder.Constant(new FloatAttribute(2.0f), "c1");
        var const2 = _builder.Constant(new FloatAttribute(3.0f), "c2");
        var result = _builder.Add(const1, const2);

        func.SetResults(result);

        var pass = new ConstantFoldingPass();
        var changed = pass.Run(_module);

        Assert.IsTrue(changed);
        // Verify AddOp is replaced with ConstantOp(value=5.0)
    }

    [Test]
    public void DeadCodeEliminationPass_RemovesUnusedOperations()
    {
        var func = _module.CreateFunction("Test");
        var input = func.AddParameter(new TensorType(DataType.Float32, new[] { 1 }), "x");
        var temp = _builder.Add(input, input);
        var output = _builder.Add(temp, _builder.Constant(new FloatAttribute(1.0f)));

        // Temp is used, so should not be eliminated
        var unused = _builder.Add(input, input);  // This should be eliminated

        func.SetResults(output);

        var pass = new DeadCodeEliminationPass();
        var changed = pass.Run(_module);

        Assert.IsTrue(changed);
        Assert.AreEqual(3, func.Body.Operations.Count);  // input, temp, output
    }

    [Test]
    public void OperationSimplificationPass_SimplifiesIdentityOperations()
    {
        var func = _module.CreateFunction("Test");
        var input = func.AddParameter(new TensorType(DataType.Float32, new[] { 1 }), "x");
        var zero = _builder.Constant(new FloatAttribute(0.0f));
        var result = _builder.Add(input, zero);

        func.SetResults(result);

        var pass = new OperationSimplificationPass();
        var changed = pass.Run(_module);

        Assert.IsTrue(changed);
        // Verify AddOp is replaced with just input
    }
}
```

### Lowering Pass Tests

**LoweringPassTests.cs**
```csharp
[TestFixture]
public class LoweringPassTests
{
    private HLIRModule _module;
    private HLIRBuilder _builder;

    [SetUp]
    public void Setup()
    {
        _module = new HLIRModule();
        _builder = new HLIRBuilder(_module);
    }

    [Test]
    public void HLIRtoMLIRLoweringPass_LowersSimpleGraph()
    {
        var func = _module.CreateFunction("Test");
        var input = func.AddParameter(new TensorType(DataType.Float32, new[] { 32, 64 }), "x");
        var weight = _builder.Constant(new TensorAttribute(...), "w");
        var output = _builder.MatMul(input, weight);

        func.SetResults(output);

        var pass = new HLIRtoMLIRLoweringPass();
        var changed = pass.Run(_module);

        Assert.IsTrue(changed);
        // Verify operations are now MLIR ops
    }

    [Test]
    public void MLIRtoLLIRLoweringPass_LowersToLowLevel()
    {
        // First lower to MLIR
        var hlirModule = _module;
        var hlirToMlir = new HLIRtoMLIRLoweringPass();
        hlirToMlir.Run(hlirModule);

        // Then lower to LLIR
        var mlirToLlir = new MLIRtoLLIRLoweringPass();
        var changed = mlirToLlir.Run(hlirModule);

        Assert.IsTrue(changed);
        // Verify operations include explicit memory ops
    }
}
```

### IR Printer Tests

**IRPrinterTests.cs**
```csharp
[TestFixture]
public class IRPrinterTests
{
    [Test]
    public void TextIRPrinter_PrintsModule()
    {
        var module = new HLIRModule();
        var builder = new HLIRBuilder(module);
        var func = module.CreateFunction("Test");
        var input = func.AddParameter(new TensorType(DataType.Float32, new[] { 1 }), "x");
        var output = _builder.Add(input, input);
        func.SetResults(output);

        var printer = new TextIRPrinter();
        var output = printer.Print(module);

        Assert.IsNotNull(output);
        Assert.IsTrue(output.Contains("module"));
        Assert.IsTrue(output.Contains("function Test"));
        Assert.IsTrue(output.Contains("add"));
    }

    [Test]
    public void IRDebugger_FindsOperations()
    {
        var module = new HLIRModule();
        var builder = new HLIRBuilder(module);
        var func = module.CreateFunction("Test");
        var input = func.AddParameter(new TensorType(DataType.Float32, new[] { 1 }), "x");
        var output = _builder.Add(input, input);
        func.SetResults(output);

        var debugger = new IRDebugger(module);
        var op = debugger.FindOperation(output.Id);

        Assert.IsNotNull(op);
        Assert.IsInstanceOf<AddOp>(op);
    }
}
```

## Deliverables

- `tests/IR/Types/TensorTypeTests.cs`
- `tests/IR/Operations/AddOpTests.cs`
- `tests/IR/Operations/MatMulOpTests.cs`
- `tests/IR/Operations/Conv2DOpTests.cs`
- `tests/IR/Graph/GraphBuilderTests.cs`
- `tests/IR/Passes/OptimizationPassTests.cs`
- `tests/IR/Lowering/LoweringPassTests.cs`
- `tests/IR/Debug/IRPrinterTests.cs`

## Success Criteria

- All tests pass
- Test coverage > 80% for core IR functionality
- Tests verify correctness of type system, operations, graph building, and transformations

## Dependencies

- All previous specs (IR type system, HLIR, MLIR, LLIR, transformations, etc.)
