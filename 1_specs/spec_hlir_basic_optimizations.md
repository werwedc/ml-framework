# Spec: Basic High-Level IR Optimization Passes

## Overview
Implement fundamental optimization passes that operate on High-Level IR. These passes improve the IR representation before lowering to MLIR, removing unnecessary operations and simplifying the graph.

## Requirements

### Constant Folding Pass

```csharp
public class ConstantFoldingPass : IRTransformation
{
    public ConstantFoldingPass() : base("ConstantFolding", false) { }

    public override bool Run(HLIRModule module)
    {
        bool changed = false;
        var evaluator = new ConstantEvaluator();

        foreach (var function in module.Functions)
        {
            changed |= FoldConstants(function, evaluator);
        }

        return changed;
    }

    private bool FoldConstants(HLIRFunction function, ConstantEvaluator evaluator)
    {
        // For each operation:
        // 1. Check if all operands are constants
        // 2. If yes, evaluate the operation
        // 3. Replace operation with ConstantOp
        // 4. Update uses
        return false;
    }
}
```

### Constant Evaluator

```csharp
public class ConstantEvaluator
{
    public TensorValue Evaluate(IROperation op);
    private TensorValue EvaluateAdd(AddOp op);
    private TensorValue EvaluateSub(SubOp op);
    private TensorValue EvaluateMul(MulOp op);
    private TensorValue EvaluateDiv(DivOp op);
    private TensorValue EvaluateMatMul(MatMulOp op);
    private TensorValue EvaluateConstant(ConstantOp op);
}
```

### Dead Code Elimination Pass

```csharp
public class DeadCodeEliminationPass : IRTransformation
{
    public DeadCodeEliminationPass() : base("DeadCodeElimination", false) { }

    public override bool Run(HLIRModule module)
    {
        bool changed = false;

        foreach (var function in module.Functions)
        {
            changed |= EliminateDeadCode(function);
        }

        return changed;
    }

    private bool EliminateDeadCode(HLIRFunction function)
    {
        // 1. Mark all values in function.Results as live
        // 2. Recursively mark values used by live operations
        // 3. Remove all operations that don't produce live values
        return false;
    }
}
```

### Common Subexpression Elimination (CSE) Pass

```csharp
public class CSEPass : IRTransformation
{
    private struct OperationKey
    {
        public IROpcode Opcode;
        public IRValue[] Operands;
        public int GetHashCode();
        public bool Equals(object obj);
    }

    public CSEPass() : base("CommonSubexpressionElimination", false) { }

    public override bool Run(HLIRModule module)
    {
        bool changed = false;

        foreach (var function in module.Functions)
        {
            changed |= EliminateCommonSubexpressions(function);
        }

        return changed;
    }

    private bool EliminateCommonSubexpressions(HLIRFunction function)
    {
        // 1. Build hash map of operation -> result value
        // 2. For each operation, check if equivalent operation exists
        // 3. If yes, replace result with existing result
        // 4. Remove duplicate operation
        return false;
    }
}
```

### Operation Simplification Pass

```csharp
public class OperationSimplificationPass : IRTransformation
{
    public OperationSimplificationPass() : base("OperationSimplification", false) { }

    public override bool Run(HLIRModule module)
    {
        bool changed = false;

        foreach (var function in module.Functions)
        {
            changed |= SimplifyOperations(function);
        }

        return changed;
    }

    private bool SimplifyOperations(HLIRFunction function)
    {
        // Simplify patterns like:
        // - x + 0 -> x
        // - x * 1 -> x
        // - x * 0 -> 0
        // - identity transpose/reshape
        // - redundant broadcasts
        return false;
    }

    private bool SimplifyAddOp(AddOp op)
    {
        // x + 0 -> x
        // 0 + x -> x
        return false;
    }

    private bool SimplifyMulOp(MulOp op)
    {
        // x * 1 -> x
        // 1 * x -> x
        // x * 0 -> 0 (if x is not NaN)
        return false;
    }

    private bool SimplifyReshape(ReshapeOp op)
    {
        // Remove identity reshapes
        return false;
    }

    private bool SimplifyTranspose(TransposeOp op)
    {
        // Remove identity transposes
        // Combine consecutive transposes
        return false;
    }
}
```

### Inline Constants Pass

```csharp
public class InlineConstantsPass : IRTransformation
{
    public InlineConstantsPass() : base("InlineConstants", false) { }

    public override bool Run(HLIRModule module)
    {
        bool changed = false;

        foreach (var function in module.Functions)
        {
            changed |= InlineConstants(function);
        }

        return changed;
    }

    private bool InlineConstants(HLIRFunction function)
    {
        // For small constant tensors (< some threshold):
        // 1. Load the constant value
        // 2. Inline it directly into operations
        // 3. Remove the ConstantOp
        return false;
    }
}
```

## Implementation Details

1. **Pass Ordering**: These passes should typically run in order: InlineConstants → Simplification → CSE → ConstantFolding → DCE
2. **Idempotency**: Each pass should be idempotent (running it multiple times has no effect after the first)
3. **Change Tracking**: Passes should track if they made changes to avoid unnecessary re-runs
4. **Analysis Results**: Some passes (like DCE) can use analysis from other passes

## Deliverables

- `src/IR/Passes/ConstantFoldingPass.cs`
- `src/IR/Passes/ConstantEvaluator.cs`
- `src/IR/Passes/DeadCodeEliminationPass.cs`
- `src/IR/Passes/CSEPass.cs`
- `src/IR/Passes/OperationSimplificationPass.cs`
- `src/IR/Passes/InlineConstantsPass.cs`

## Success Criteria

- ConstantFolding: Evaluates and replaces constant expressions
- DCE: Removes all unused operations
- CSE: Eliminates duplicate computations
- Simplification: Reduces trivial operations
- All passes report whether they made changes

## Dependencies

- spec_ir_type_system.md
- spec_hlir_operations.md
- spec_hlir_graph_builder.md
- spec_ir_transformation_infra.md
