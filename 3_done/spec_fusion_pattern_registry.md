# Spec: Fusion Pattern Registry

## Overview
Implement a registry system for defining and managing fusible operation types and fusion patterns. This centralizes fusion logic and allows extensibility.

## Requirements

### 1. Fusion Registry Interface
Core registry for managing fusible operations and patterns.

```csharp
public interface IFusionRegistry
{
    /// <summary>
    /// Registers an operation as fusible with specific constraints
    /// </summary>
    void RegisterFusibleOperation(string opType, FusibleOpConstraints constraints);

    /// <summary>
    /// Registers a composite fusion pattern
    /// </summary>
    void RegisterFusionPattern(string patternName, FusionPatternDefinition pattern);

    /// <summary>
    /// Gets all registered fusible operation types
    /// </summary>
    IReadOnlySet<string> GetFusibleOperations();

    /// <summary>
    /// Gets pattern definition by name
    /// </summary>
    FusionPatternDefinition? GetPattern(string patternName);

    /// <summary>
    /// Finds applicable patterns for a sequence of operations
    /// </summary>
    List<FusionPatternMatch> FindMatches(IEnumerable<Operation> operations);
}

public record FusibleOpConstraints
{
    public required TensorLayout RequiredLayout { get; init; }
    public required IReadOnlySet<TensorDataType> SupportedDataTypes { get; init; }
    public bool RequiresContiguousMemory { get; init; } = true;
    public int MaxFusionGroupSize { get; init; } = 16;
    public bool SupportsFusionWithInplaceOps { get; init; } = false;
}

public record FusionPatternDefinition
{
    public required string Name { get; init; }
    public required IReadOnlyList<string> OpTypeSequence { get; init; }
    public required PatternMatchDelegate MatchStrategy { get; init; }
    public FusionStrategy Strategy { get; init; } = FusionStrategy.Merge;
    public int Priority { get; init; } = 0;
}

public record FusionPatternMatch
{
    public required FusionPatternDefinition Pattern { get; init; }
    public required IReadOnlyList<Operation> MatchedOperations { get; init; }
    public required int MatchScore { get; init; }
}

public delegate bool PatternMatchDelegate(IReadOnlyList<Operation> operations);

public enum FusionStrategy
{
    Merge,           // Merge operations into single kernel
    Fold,           // Fold parameters (e.g., BN into Conv)
    Replace,        // Replace with specialized kernel
    Inplace         // Perform operations in-place
}
```

### 2. Default Pattern Registry
Pre-registered common fusion patterns.

```csharp
public class DefaultFusionRegistry : IFusionRegistry
{
    private readonly Dictionary<string, FusibleOpConstraints> _fusibleOps = new();
    private readonly Dictionary<string, FusionPatternDefinition> _patterns = new();
    private readonly List<FusionPatternDefinition> _orderedPatterns = new();

    public DefaultFusionRegistry()
    {
        RegisterDefaultElementWiseOperations();
        RegisterDefaultFusionPatterns();
    }

    private void RegisterDefaultElementWiseOperations()
    {
        // Register element-wise operations
        RegisterFusibleOperation("Add", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<TensorDataType> {
                TensorDataType.Float32, TensorDataType.Float16,
                TensorDataType.Int32, TensorDataType.Int64
            },
            SupportsFusionWithInplaceOps = true
        });

        RegisterFusibleOperation("Mul", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<TensorDataType> {
                TensorDataType.Float32, TensorDataType.Float16,
                TensorDataType.Int32, TensorDataType.Int64
            },
            SupportsFusionWithInplaceOps = true
        });

        RegisterFusibleOperation("ReLU", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<TensorDataType> {
                TensorDataType.Float32, TensorDataType.Float16
            }
        });

        RegisterFusibleOperation("Sigmoid", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<TensorDataType> {
                TensorDataType.Float32, TensorDataType.Float16
            }
        });

        RegisterFusibleOperation("Tanh", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<TensorDataType> {
                TensorDataType.Float32, TensorDataType.Float16
            }
        });

        RegisterFusibleOperation("LeakyReLU", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<TensorDataType> {
                TensorDataType.Float32, TensorDataType.Float16
            }
        });
    }

    private void RegisterDefaultFusionPatterns()
    {
        // Element-wise chain pattern
        RegisterFusionPattern("ElementWiseChain", new FusionPatternDefinition
        {
            Name = "ElementWiseChain",
            OpTypeSequence = new[] { "Add", "Mul", "ReLU", "Sigmoid", "Tanh" },
            MatchStrategy = MatchElementWiseChain,
            Strategy = FusionStrategy.Merge,
            Priority = 10
        });

        // Conv + Activation pattern
        RegisterFusionPattern("ConvActivation", new FusionPatternDefinition
        {
            Name = "ConvActivation",
            OpTypeSequence = new[] { "Conv2D", "ReLU" },
            MatchStrategy = MatchConvActivation,
            Strategy = FusionStrategy.Merge,
            Priority = 20
        });

        // Conv + BatchNorm pattern
        RegisterFusionPattern("ConvBatchNorm", new FusionPatternDefinition
        {
            Name = "ConvBatchNorm",
            OpTypeSequence = new[] { "Conv2D", "BatchNorm" },
            MatchStrategy = MatchConvBatchNorm,
            Strategy = FusionStrategy.Fold,
            Priority = 25
        });

        // Linear + Activation pattern
        RegisterFusionPattern("LinearActivation", new FusionPatternDefinition
        {
            Name = "LinearActivation",
            OpTypeSequence = new[] { "Linear", "ReLU" },
            MatchStrategy = MatchLinearActivation,
            Strategy = FusionStrategy.Merge,
            Priority = 20
        });
    }
}
```

### 3. Pattern Matching Strategies
Implement pattern matching delegates.

```csharp
public static class PatternMatchers
{
    public static bool MatchElementWiseChain(IReadOnlyList<Operation> operations)
    {
        if (operations.Count < 2)
            return false;

        // All operations must be element-wise
        foreach (var op in operations)
        {
            if (!IsElementWiseOperation(op.Type))
                return false;
        }

        // Check shape compatibility
        for (int i = 1; i < operations.Count; i++)
        {
            if (!ShapesCompatible(operations[i - 1], operations[i]))
                return false;
        }

        return true;
    }

    public static bool MatchConvActivation(IReadOnlyList<Operation> operations)
    {
        if (operations.Count != 2)
            return false;

        return operations[0].Type == "Conv2D" &&
               IsActivationOperation(operations[1].Type) &&
               ConvActivationCompatible(operations[0], operations[1]);
    }

    public static bool MatchConvBatchNorm(IReadOnlyList<Operation> operations)
    {
        if (operations.Count != 2)
            return false;

        return operations[0].Type == "Conv2D" &&
               operations[1].Type == "BatchNorm" &&
               BatchNormFoldable(operations[1]);
    }

    public static bool MatchLinearActivation(IReadOnlyList<Operation> operations)
    {
        if (operations.Count != 2)
            return false;

        return operations[0].Type == "Linear" &&
               IsActivationOperation(operations[1].Type);
    }

    private static bool IsElementWiseOperation(string opType)
    {
        return opType is "Add" or "Sub" or "Mul" or "Div" or
               "ReLU" or "Sigmoid" or "Tanh" or "LeakyReLU" or
               "Exp" or "Log" or "Abs" or "Neg";
    }

    private static bool IsActivationOperation(string opType)
    {
        return opType is "ReLU" or "Sigmoid" or "Tanh" or "LeakyReLU" or
               "GELU" or "Swish" or "Softmax";
    }

    private static bool ShapesCompatible(Operation op1, Operation op2)
    {
        // Element-wise ops should maintain same shape
        return op1.OutputShape.Equals(op2.InputShape);
    }
}
```

### 4. Registry Factory
Factory for creating and configuring registries.

```csharp
public static class FusionRegistryFactory
{
    public static IFusionRegistry CreateDefault()
    {
        return new DefaultFusionRegistry();
    }

    public static IFusionRegistry CreateWithCustomPatterns(
        Action<IFusionRegistry> configure)
    {
        var registry = new DefaultFusionRegistry();
        configure(registry);
        return registry;
    }

    public static IFusionRegistry CreateEmpty()
    {
        return new DefaultFusionRegistry(skipDefaults: true);
    }
}
```

## Implementation Tasks

1. **Create registry interfaces and data structures** (20 min)
   - IFusionRegistry interface
   - FusibleOpConstraints record
   - FusionPatternDefinition record
   - FusionPatternMatch record
   - FusionStrategy enum

2. **Implement DefaultFusionRegistry core** (25 min)
   - Dictionary-based storage
   - Operation registration logic
   - Pattern registration logic
   - Priority-based pattern ordering

3. **Register default element-wise operations** (15 min)
   - Add, Mul, Sub, Div
   - ReLU, Sigmoid, Tanh
   - Exp, Log, Abs, Neg

4. **Register default fusion patterns** (20 min)
   - ElementWiseChain pattern
   - ConvActivation pattern
   - ConvBatchNorm pattern
   - LinearActivation pattern

5. **Implement pattern matching strategies** (25 min)
   - MatchElementWiseChain
   - MatchConvActivation
   - MatchConvBatchNorm
   - MatchLinearActivation
   - Helper validation methods

6. **Create registry factory** (15 min)
   - CreateDefault()
   - CreateWithCustomPatterns()
   - CreateEmpty()

## Test Cases

```csharp
[Test]
public void RegisterOperation_Retrievable()
{
    var registry = new DefaultFusionRegistry();
    registry.RegisterFusibleOperation("CustomOp", new FusibleOpConstraints
    {
        RequiredLayout = TensorLayout.NCHW,
        SupportedDataTypes = new HashSet<TensorDataType> { TensorDataType.Float32 }
    });

    var fusibleOps = registry.GetFusibleOperations();
    Assert.IsTrue(fusibleOps.Contains("CustomOp"));
}

[Test]
public void FindPattern_MatchesConvReLU()
{
    var registry = new DefaultFusionRegistry();
    var ops = new[] { CreateConvOp(), CreateReluOp() };

    var matches = registry.FindMatches(ops);

    Assert.AreEqual(1, matches.Count);
    Assert.AreEqual("ConvActivation", matches[0].Pattern.Name);
}

[Test]
public void FindPattern_PrioritizesHigherPriority()
{
    var registry = new DefaultFusionRegistry();
    // Register higher priority pattern
    // Find matches and verify priority ordering
}

[Test]
public void MatchElementWiseChain_RejectsNonElementWise()
{
    var ops = new[] { CreateAddOp(), CreateConvOp(), CreateReluOp() };

    var result = PatternMatchers.MatchElementWiseChain(ops);

    Assert.IsFalse(result);
}
```

## Success Criteria
- Registry correctly stores and retrieves fusible operations
- Pattern matching accurately identifies fusion candidates
- Priority-based pattern selection works correctly
- Custom patterns can be registered dynamically

## Dependencies
- Operation abstraction
- Tensor data types
- Tensor layout types
