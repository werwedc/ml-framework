# Spec: IR Transformation/Lowering Infrastructure

## Overview
Define the infrastructure for transforming IR between different levels (HLIR → MLIR → LLIR → Backend IR). This provides the base classes and interfaces for all lowering passes and transformations.

## Requirements

### IRTransformation Base Class

```csharp
public abstract class IRTransformation
{
    public string Name { get; }
    public bool IsAnalysisOnly { get; }

    protected IRTransformation(string name, bool isAnalysisOnly = false);

    public abstract bool Run(HLIRModule module);
    public virtual void Initialize(HLIRModule module) { }
    public virtual void Cleanup() { }
}
```

### Lowering Pass Interface

```csharp
public interface ILoweringPass
{
    string SourceIRLevel { get; }
    string TargetIRLevel { get; }

    bool CanLower(IROperation op);
    IROperation Lower(IRContext targetContext, IROperation op);
}

public abstract class LoweringPassBase : IRTransformation, ILoweringPass
{
    protected LoweringPassBase(string sourceLevel, string targetLevel)
        : base($"{sourceLevel}To{targetLevel}", false) { }

    public abstract bool CanLower(IROperation op);
    public abstract IROperation Lower(IRContext targetContext, IROperation op);

    public override bool Run(HLIRModule module)
    {
        bool changed = false;
        var targetContext = new IRContext();

        // Lower all operations
        foreach (var function in module.Functions)
        {
            changed |= LowerFunction(function, targetContext);
        }

        return changed;
    }

    protected virtual bool LowerFunction(HLIRFunction function, IRContext targetContext)
    {
        // Implementation to be provided by derived classes
        return false;
    }
}
```

### PassManager

```csharp
public class IRPassManager
{
    public enum PassType
    {
        Analysis,
        Optimization,
        Lowering,
        Validation
    }

    private List<IRTransformation> _passes;
    private Dictionary<PassType, List<IRTransformation>> _passesByType;

    public void AddPass(IRTransformation pass, PassType type = PassType.Optimization);
    public bool RunAll(HLIRModule module);
    public bool RunAnalysisPasses(HLIRModule module);
    public bool RunOptimizationPasses(HLIRModule module);
    public bool RunLoweringPasses(HLIRModule module);
    public bool RunValidationPasses(HLIRModule module);
}
```

### Operation Rewriter

```csharp
public class OperationRewriter
{
    private IRContext _sourceContext;
    private IRContext _targetContext;
    private Dictionary<IRValue, IRValue> _valueMap;

    public OperationRewriter(IRContext source, IRContext target);
    public IRValue RemapValue(IRValue value);
    public void SetMapping(IRValue source, IRValue target);

    public IRBlock RemapBlock(IRBlock sourceBlock, IRContext targetContext);
    public HLIRFunction RemapFunction(HLIRFunction sourceFunction, IRContext targetContext);
}
```

### IR Verifier

```csharp
public class IRVerifier : IRTransformation
{
    private List<string> _errors;
    private List<string> _warnings;

    public IRVerifier() : base("Verifier", true) { }

    public List<string> Errors => _errors;
    public List<string> Warnings => _warnings;

    public override bool Run(HLIRModule module)
    {
        _errors.Clear();
        _warnings.Clear();

        foreach (var function in module.Functions)
        {
            VerifyFunction(function);
        }

        return _errors.Count == 0;
    }

    private void VerifyFunction(HLIRFunction function)
    {
        // Check all values are defined before use
        // Check all operations are valid
        // Check types match
        // Check no cycles in data flow
    }
}
```

### Pass Instrumentation

```csharp
public class PassInstrumentation
{
    public event Action<IRTransformation, HLIRModule> BeforePass;
    public event Action<IRTransformation, HLIRModule> AfterPass;
    public event Action<IRTransformation, HLIRModule, Exception> OnPassError;

    public void NotifyBeforePass(IRTransformation pass, HLIRModule module);
    public void NotifyAfterPass(IRTransformation pass, HLIRModule module);
    public void NotifyPassError(IRTransformation pass, HLIRModule module, Exception ex);
}
```

## Implementation Details

1. **Value Mapping**: `OperationRewriter` maintains a mapping from source to target values during lowering
2. **Pass Dependencies**: `IRPassManager` should support pass dependency specification
3. **Caching**: Caching for analysis results to avoid redundant computation
4. **Error Reporting**: Collect all errors before failing to provide comprehensive feedback

## Deliverables

- `src/IR/Transformations/IRTransformation.cs`
- `src/IR/Transformations/ILoweringPass.cs`
- `src/IR/Transformations/LoweringPassBase.cs`
- `src/IR/Transformations/IRPassManager.cs`
- `src/IR/Transformations/OperationRewriter.cs`
- `src/IR/Transformations/IRVerifier.cs`
- `src/IR/Transformations/PassInstrumentation.cs`

## Success Criteria

- Can create and register custom transformations
- PassManager executes passes in order
- OperationRewriter correctly remaps values and blocks
- IRVerifier catches common IR errors
- Pass instrumentation events fire correctly

## Dependencies

- spec_ir_type_system.md
- spec_hlir_operations.md
- spec_hlir_graph_builder.md
