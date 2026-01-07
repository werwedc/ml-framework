# Spec: Fusion Validation and Constraints

## Overview
Implement comprehensive validation system for fusion transformations, including compatibility checks, constraint validation, and fallback mechanisms.

## Requirements

### 1. Fusion Constraints System
Define and validate fusion constraints.

```csharp
public interface IFusionConstraints
{
    /// <summary>
    /// Validates that operations satisfy fusion constraints
    /// </summary>
    bool Satisfies(IReadOnlyList<Operation> operations);

    /// <summary>
    /// Gets detailed constraint violations
    /// </summary>
    IReadOnlyList<ConstraintViolation> GetViolations(IReadOnlyList<Operation> operations);
}

public record ConstraintViolation
{
    public required string ConstraintName { get; init; }
    public required string Message { get; init; }
    public required Severity Severity { get; init; }
}

public enum Severity
{
    Error,   // Cannot fuse
    Warning  // Can fuse but with caveats
}

public class FusionConstraintsValidator
{
    private readonly List<IFusionConstraints> _constraints;

    public FusionConstraintsValidator()
    {
        _constraints = new List<IFusionConstraints>
        {
            new MemoryLayoutConstraint(),
            new NumericalPrecisionConstraint(),
            new ThreadBlockConstraint(),
            new SideEffectConstraint(),
            new ControlFlowConstraint(),
            new MemoryAccessPatternConstraint()
        };
    }

    public bool Validate(IReadOnlyList<Operation> operations, out IReadOnlyList<ConstraintViolation> violations)
    {
        var allViolations = new List<ConstraintViolation>();

        foreach (var constraint in _constraints)
        {
            var constraintViolations = constraint.GetViolations(operations);
            allViolations.AddRange(constraintViolations);
        }

        violations = allViolations;
        return !violations.Any(v => v.Severity == Severity.Error);
    }
}
```

### 2. Individual Constraint Validators
Implement specific constraint validators.

**Memory Layout Constraint:**
```csharp
public class MemoryLayoutConstraint : IFusionConstraints
{
    public bool Satisfies(IReadOnlyList<Operation> operations)
    {
        return !GetViolations(operations).Any(v => v.Severity == Severity.Error);
    }

    public IReadOnlyList<ConstraintViolation> GetViolations(IReadOnlyList<Operation> operations)
    {
        var violations = new List<ConstraintViolation>();

        if (operations.Count == 0)
            return violations;

        var referenceLayout = operations[0].Layout;

        for (int i = 1; i < operations.Count; i++)
        {
            var opLayout = operations[i].Layout;

            if (opLayout != referenceLayout && opLayout != TensorLayout.Any && referenceLayout != TensorLayout.Any)
            {
                violations.Add(new ConstraintViolation
                {
                    ConstraintName = "MemoryLayout",
                    Message = $"Operation {i} has layout {opLayout}, expected {referenceLayout}",
                    Severity = Severity.Error
                });
            }
        }

        return violations;
    }
}
```

**Numerical Precision Constraint:**
```csharp
public class NumericalPrecisionConstraint : IFusionConstraints
{
    public bool Satisfies(IReadOnlyList<Operation> operations)
    {
        return !GetViolations(operations).Any(v => v.Severity == Severity.Error);
    }

    public IReadOnlyList<ConstraintViolation> GetViolations(IReadOnlyList<Operation> operations)
    {
        var violations = new List<ConstraintViolation>();

        if (operations.Count == 0)
            return violations;

        var referenceDtype = operations[0].DataType;

        for (int i = 1; i < operations.Count; i++)
        {
            var opDtype = operations[i].DataType;

            if (opDtype != referenceDtype)
            {
                violations.Add(new ConstraintViolation
                {
                    ConstraintName = "NumericalPrecision",
                    Message = $"Operation {i} has dtype {opDtype}, expected {referenceDtype}",
                    Severity = Severity.Error
                });
            }
        }

        // Check for precision-sensitive operations
        var hasPrecisionSensitiveOps = operations.Any(op =>
            op.Type == "ReduceSum" || op.Type == "ReduceMean");

        if (hasPrecisionSensitiveOps && referenceDtype == TensorDataType.Float16)
        {
            violations.Add(new ConstraintViolation
            {
                ConstraintName = "NumericalPrecision",
                Message = "Precision-sensitive operations with FP16 may cause numerical instability",
                Severity = Severity.Warning
            });
        }

        return violations;
    }
}
```

**Thread Block Configuration Constraint:**
```csharp
public class ThreadBlockConstraint : IFusionConstraints
{
    public bool Satisfies(IReadOnlyList<Operation> operations)
    {
        return !GetViolations(operations).Any(v => v.Severity == Severity.Error);
    }

    public IReadOnlyList<ConstraintViolation> GetViolations(IReadOnlyList<Operation> operations)
    {
        var violations = new List<ConstraintViolation>();

        // Calculate required thread configuration
        var requiredThreads = CalculateRequiredThreads(operations);
        var maxThreads = GetMaxThreadsPerBlock();

        if (requiredThreads > maxThreads)
        {
            violations.Add(new ConstraintViolation
            {
                ConstraintName = "ThreadBlock",
                Message = $"Requires {requiredThreads} threads, exceeds maximum {maxThreads}",
                Severity = Severity.Error
            });
        }

        // Check for incompatible thread configurations
        var threadConfigs = operations.Select(op => op.ThreadBlockConfig).Distinct().ToList();
        if (threadConfigs.Count > 1)
        {
            violations.Add(new ConstraintViolation
            {
                ConstraintName = "ThreadBlock",
                Message = "Operations have incompatible thread block configurations",
                Severity = Severity.Error
            });
        }

        return violations;
    }

    private int CalculateRequiredThreads(IReadOnlyList<Operation> operations)
    {
        // Estimate based on output tensor size and operation type
        var outputShape = operations[^1].OutputShape;
        return (int)(outputShape.Width * outputShape.Height);
    }

    private int GetMaxThreadsPerBlock()
    {
        // This would come from device capabilities
        return 1024;
    }
}
```

**Side Effect Constraint:**
```csharp
public class SideEffectConstraint : IFusionConstraints
{
    public bool Satisfies(IReadOnlyList<Operation> operations)
    {
        return !GetViolations(operations).Any(v => v.Severity == Severity.Error);
    }

    public IReadOnlyList<ConstraintViolation> GetViolations(IReadOnlyList<Operation> operations)
    {
        var violations = new List<ConstraintViolation>();

        foreach (var op in operations)
        {
            if (HasSideEffects(op))
            {
                violations.Add(new ConstraintViolation
                {
                    ConstraintName = "SideEffect",
                    Message = $"Operation {op.Name} ({op.Type}) has side effects",
                    Severity = Severity.Error
                });
            }
        }

        return violations;
    }

    private bool HasSideEffects(Operation op)
    {
        // Operations with external side effects
        return op.Type is "Print" or "WriteToFile" or "Send";
    }
}
```

**Control Flow Constraint:**
```csharp
public class ControlFlowConstraint : IFusionConstraints
{
    public bool Satisfies(IReadOnlyList<Operation> operations)
    {
        return !GetViolations(operations).Any(v => v.Severity == Severity.Error);
    }

    public IReadOnlyList<ConstraintViolation> GetViolations(IReadOnlyList<Operation> operations)
    {
        var violations = new List<ConstraintViolation>();

        foreach (var op in operations)
        {
            if (HasDataDependentControlFlow(op))
            {
                violations.Add(new ConstraintViolation
                {
                    ConstraintName = "ControlFlow",
                    Message = $"Operation {op.Name} has data-dependent control flow",
                    Severity = Severity.Error
                });
            }

            if (HasComplexBranching(op))
            {
                violations.Add(new ConstraintViolation
                {
                    ConstraintName = "ControlFlow",
                    Message = $"Operation {op.Name} has complex branching",
                    Severity = Severity.Error
                });
            }
        }

        return violations;
    }

    private bool HasDataDependentControlFlow(Operation op)
    {
        // Operations with dynamic control based on data
        return op.Type == "Where" || op.Type == "DynamicIf";
    }

    private bool HasComplexBranching(Operation op)
    {
        // Operations with complex internal branching
        return op.Type == "Loop" || op.Type == "Recursion";
    }
}
```

**Memory Access Pattern Constraint:**
```csharp
public class MemoryAccessPatternConstraint : IFusionConstraints
{
    public bool Satisfies(IReadOnlyList<Operation> operations)
    {
        return !GetViolations(operations).Any(v => v.Severity == Severity.Error);
    }

    public IReadOnlyList<ConstraintViolation> GetViolations(IReadOnlyList<Operation> operations)
    {
        var violations = new List<ConstraintViolation>();

        if (operations.Count == 0)
            return violations;

        var analyzer = new GraphAnalyzer();
        var patterns = operations.Select(op => analyzer.AnalyzeAccessPattern(op)).ToList();

        // Check for incompatible access patterns
        for (int i = 1; i < patterns.Count; i++)
        {
            if (!ArePatternsCompatible(patterns[i - 1], patterns[i]))
            {
                violations.Add(new ConstraintViolation
                {
                    ConstraintName = "MemoryAccessPattern",
                    Message = $"Incompatible access patterns: {patterns[i-1]} and {patterns[i]}",
                    Severity = Severity.Error
                });
            }
        }

        // Check for gather/scatter operations (hard to fuse)
        if (patterns.Any(p => p == MemoryAccessPattern.Gather || p == MemoryAccessPattern.Scatter))
        {
            violations.Add(new ConstraintViolation
            {
                ConstraintName = "MemoryAccessPattern",
                Message = "Gather/Scatter operations are difficult to fuse efficiently",
                Severity = Severity.Warning
            });
        }

        return violations;
    }

    private bool ArePatternsCompatible(MemoryAccessPattern p1, MemoryAccessPattern p2)
    {
        // Define compatible pattern combinations
        var compatiblePairs = new HashSet<(MemoryAccessPattern, MemoryAccessPattern)>
        {
            (MemoryAccessPattern.ElementWise, MemoryAccessPattern.ElementWise),
            (MemoryAccessPattern.ElementWise, MemoryAccessPattern.Reduction),
            (MemoryAccessPattern.Spatial, MemoryAccessPattern.ElementWise)
        };

        return compatiblePairs.Contains((p1, p2)) || compatiblePairs.Contains((p2, p1));
    }
}
```

### 3. Fallback Mechanism
Handle cases where fusion cannot be applied.

```csharp
public interface IFusionFallback
{
    /// <summary>
    /// Attempts to execute operations as separate kernels
    /// </summary>
    Tensor ExecuteSeparate(IReadOnlyList<Operation> operations, Tensor input);

    /// <summary>
    /// Logs the reason for falling back
    /// </summary>
    void LogFallbackReason(string reason, IReadOnlyList<Operation> operations);
}

public class FusionFallbackHandler : IFusionFallback
{
    private readonly ILogger _logger;
    private readonly IKernelExecutor _executor;

    public FusionFallbackHandler(ILogger logger, IKernelExecutor executor)
    {
        _logger = logger;
        _executor = executor;
    }

    public Tensor ExecuteSeparate(IReadOnlyList<Operation> operations, Tensor input)
    {
        Tensor output = input;

        foreach (var op in operations)
        {
            output = _executor.ExecuteKernel(op, output);
        }

        return output;
    }

    public void LogFallbackReason(string reason, IReadOnlyList<Operation> operations)
    {
        var opTypes = string.Join(" -> ", operations.Select(op => op.Type));
        _logger.LogWarning(
            "Fusion fallback for chain: {OpChain}. Reason: {Reason}",
            opTypes, reason);
    }
}
```

### 4. Fusion Verification System
Verify that fused kernels produce correct results.

```csharp
public interface IFusionVerifier
{
    /// <summary>
    /// Verifies that fused operation produces same result as sequential ops
    /// </summary>
    VerificationResult Verify(
        FusedOperation fusedOp,
        IReadOnlyList<Operation> originalOps,
        Tensor testInput);

    /// <summary>
    /// Runs verification on random test inputs
    /// </summary>
    VerificationResult VerifyWithRandomInputs(
        FusedOperation fusedOp,
        IReadOnlyList<Operation> originalOps,
        int testCases = 10);
}

public record VerificationResult
{
    public required bool Passed { get; init; }
    public required double MaxError { get; init; }
    public required double MeanError { get; init; }
    public required IReadOnlyList<VerificationTestResult> TestCases { get; init; }
}

public record VerificationTestResult
{
    public required int TestCaseNumber { get; init; }
    public required Tensor FusedOutput { get; init; }
    public required Tensor SequentialOutput { get; init; }
    public required double Error { get; init; }
    public required bool TolerancePassed { get; init; }
}

public class FusionVerifier : IFusionVerifier
{
    private readonly IKernelExecutor _executor;
    private readonly ITensorGenerator _generator;
    private readonly double _tolerance = 1e-5;

    public FusionVerifier(IKernelExecutor executor, ITensorGenerator generator)
    {
        _executor = executor;
        _generator = generator;
    }

    public VerificationResult Verify(
        FusedOperation fusedOp,
        IReadOnlyList<Operation> originalOps,
        Tensor testInput)
    {
        // Execute fused operation
        var fusedOutput = _executor.ExecuteFusedKernel(fusedOp, testInput);

        // Execute sequential operations
        var sequentialOutput = testInput;
        foreach (var op in originalOps)
        {
            sequentialOutput = _executor.ExecuteKernel(op, sequentialOutput);
        }

        // Compare results
        var error = ComputeError(fusedOutput, sequentialOutput);
        var tolerancePassed = error < _tolerance;

        return new VerificationResult
        {
            Passed = tolerancePassed,
            MaxError = error,
            MeanError = error,
            TestCases = new[]
            {
                new VerificationTestResult
                {
                    TestCaseNumber = 1,
                    FusedOutput = fusedOutput,
                    SequentialOutput = sequentialOutput,
                    Error = error,
                    TolerancePassed = tolerancePassed
                }
            }
        };
    }

    public VerificationResult VerifyWithRandomInputs(
        FusedOperation fusedOp,
        IReadOnlyList<Operation> originalOps,
        int testCases = 10)
    {
        var testResults = new List<VerificationTestResult>();
        var maxError = 0.0;
        var totalError = 0.0;

        for (int i = 0; i < testCases; i++)
        {
            var testInput = _generator.GenerateRandomTensor(
                fusedOp.InputShape,
                fusedOp.DataType);

            var result = Verify(fusedOp, originalOps, testInput);
            var testCase = result.TestCases[0];

            testResults.Add(testCase with { TestCaseNumber = i + 1 });
            maxError = Math.Max(maxError, testCase.Error);
            totalError += testCase.Error;
        }

        return new VerificationResult
        {
            Passed = maxError < _tolerance,
            MaxError = maxError,
            MeanError = totalError / testCases,
            TestCases = testResults
        };
    }

    private double ComputeError(Tensor a, Tensor b)
    {
        // Compute L2 norm or max absolute error
        double maxError = 0;
        var elements = a.Shape.TotalElements;

        for (int i = 0; i < elements; i++)
        {
            var diff = Math.Abs(a.GetFloat(i) - b.GetFloat(i));
            maxError = Math.Max(maxError, diff);
        }

        return maxError;
    }
}
```

## Implementation Tasks

1. **Create constraint system interfaces** (15 min)
   - IFusionConstraints interface
   - ConstraintViolation record
   - FusionConstraintsValidator

2. **Implement individual constraint validators** (40 min)
   - MemoryLayoutConstraint
   - NumericalPrecisionConstraint
   - ThreadBlockConstraint
   - SideEffectConstraint
   - ControlFlowConstraint
   - MemoryAccessPatternConstraint

3. **Implement fallback mechanism** (20 min)
   - IFusionFallback interface
   - FusionFallbackHandler
   - Separate kernel execution

4. **Implement fusion verification system** (35 min)
   - IFusionVerifier interface
   - VerificationResult and test result records
   - FusionVerifier with error computation
   - Random input testing

## Test Cases

```csharp
[Test]
public void Validate_LayoutMismatch_ReturnsError()
{
    var ops = new[]
    {
        CreateOpWithLayout(TensorLayout.NCHW),
        CreateOpWithLayout(TensorLayout.NHWC)
    };

    var validator = new FusionConstraintsValidator();
    var result = validator.Validate(ops, out var violations);

    Assert.IsFalse(result);
    Assert.IsTrue(violations.Any(v => v.ConstraintName == "MemoryLayout"));
}

[Test]
public void Validate_SideEffect_ReturnsError()
{
    var ops = new[]
    {
        CreateOpWithSideEffect("Print"),
        CreateOp("Add")
    };

    var validator = new FusionConstraintsValidator();
    var result = validator.Validate(ops, out var violations);

    Assert.IsFalse(result);
    Assert.IsTrue(violations.Any(v => v.ConstraintName == "SideEffect"));
}

[Test]
public void Verify_CorrectFusion_Passes()
{
    var fusedOp = CreateValidFusedOperation();
    var originalOps = CreateOriginalOperations();
    var testInput = CreateTestTensor();

    var verifier = new FusionVerifier(CreateExecutor(), CreateGenerator());
    var result = verifier.Verify(fusedOp, originalOps, testInput);

    Assert.IsTrue(result.Passed);
    Assert.Less(result.MaxError, 1e-5);
}

[Test]
public void Fallback_ExecutesSeparateKernels()
{
    var ops = CreateIncompatibleOperations();
    var input = CreateTestTensor();

    var fallback = new FusionFallbackHandler(CreateLogger(), CreateExecutor());
    var output = fallback.ExecuteSeparate(ops, input);

    Assert.IsNotNull(output);
}
```

## Success Criteria
- All constraint validators correctly identify incompatibilities
- Constraint violations are clearly reported with severity levels
- Fallback mechanism executes operations correctly
- Verification system detects incorrect fusions
- Random input testing catches numerical issues

## Dependencies
- Operation abstraction
- GraphAnalyzer
- IKernelExecutor (to be defined)
- ITensorGenerator (to be defined)
- ILogger (to be defined)
