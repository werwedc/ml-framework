# Spec: Fusion Pattern Detector Core

## Overview
Implement the core fusion pattern detection system that analyzes computational graphs to identify fusible operation sequences.

## Requirements

### 1. Pattern Detection Interface
Create a base interface for detecting fusible patterns in computational graphs.

```csharp
public interface IFusionPatternDetector
{
    /// <summary>
    /// Detects fusible operation sequences in the computational graph
    /// </summary>
    /// <param name="graph">Computational graph to analyze</param>
    /// <returns>List of detected fusion candidate sequences</returns>
    List<FusionCandidate> DetectPatterns(ComputationalGraph graph);

    /// <summary>
    /// Checks if a specific operation sequence is fusible
    /// </summary>
    bool IsFusible(IEnumerable<Operation> operations);
}

public record FusionCandidate
{
    public required IReadOnlyList<Operation> Operations { get; init; }
    public required FusionPatternType PatternType { get; init; }
    public required int BenefitScore { get; init; }
}

public enum FusionPatternType
{
    ElementWise,
    ReductionThenElementWise,
    ConvActivation,
    ConvBatchNorm,
    LinearActivation,
    Mixed
}
```

### 2. Graph Analyzer
Implement graph traversal and analysis utilities.

```csharp
public class GraphAnalyzer
{
    /// <summary>
    /// Builds operation dependency graph from IR
    /// </summary>
    public DependencyGraph BuildDependencyGraph(ComputationalGraph graph);

    /// <summary>
    /// Identifies sequential operation chains without branching
    /// </summary>
    public List<OperationChain> FindLinearChains(DependencyGraph graph);

    /// <summary>
    /// Analyzes memory access patterns for operations
    /// </summary>
    public MemoryAccessPattern AnalyzeAccessPattern(Operation op);
}

public record OperationChain
{
    public required IReadOnlyList<Operation> Operations { get; init; }
    public required bool HasBranching { get; init; }
}

public enum MemoryAccessPattern
{
    ElementWise,
    Spatial,
    Reduction,
    Gather,
    Scatter,
    Unknown
}
```

### 3. Basic Pattern Detectors
Implement detectors for common fusion patterns.

**Element-Wise Pattern Detector:**
```csharp
public class ElementWisePatternDetector : IFusionPatternDetector
{
    public List<FusionCandidate> DetectPatterns(ComputationalGraph graph)
    {
        // Find chains of element-wise operations
        // e.g., Add -> Mul -> ReLU -> Sigmoid
    }

    public bool IsFusible(IEnumerable<Operation> operations)
    {
        // Check all operations are element-wise
        // Compatible tensor shapes and layouts
    }
}
```

**Conv-Activation Pattern Detector:**
```csharp
public class ConvActivationPatternDetector : IFusionPatternDetector
{
    public List<FusionCandidate> DetectPatterns(ComputationalGraph graph)
    {
        // Find Conv2D -> ReLU, Conv2D -> LeakyReLU patterns
    }

    public bool IsFusible(IEnumerable<Operation> operations)
    {
        // First op must be conv/linear, second must be activation
        // Compatible parameters (stride, padding)
    }
}
```

**Conv-BatchNorm Folding Detector:**
```csharp
public class ConvBatchNormPatternDetector : IFusionPatternDetector
{
    public List<FusionCandidate> DetectPatterns(ComputationalGraph graph)
    {
        // Find Conv2D -> BatchNorm patterns for weight folding
    }

    public bool IsFusible(IEnumerable<Operation> operations)
    {
        // Validate BN parameters for folding
        // Check training vs inference mode
    }
}
```

### 4. Operation Compatibility Checker
Verify that operations can be fused together.

```csharp
public class OperationCompatibilityChecker
{
    public bool CanFuse(Operation op1, Operation op2)
    {
        return CheckMemoryLayout(op1, op2) &&
               CheckNumericalPrecision(op1, op2) &&
               CheckThreadBlockConfig(op1, op2) &&
               CheckSideEffects(op1, op2);
    }

    private bool CheckMemoryLayout(Operation op1, Operation op2)
    {
        // Both must use same layout (NHWC or NCHW)
    }

    private bool CheckNumericalPrecision(Operation op1, Operation op2)
    {
        // Both must use same dtype (FP32, FP16, etc.)
    }

    private bool CheckThreadBlockConfig(Operation op1, Operation op2)
    {
        // Compatible thread block dimensions
    }

    private bool CheckSideEffects(Operation op1, Operation op2)
    {
        // Neither should have external side effects
    }
}
```

## Implementation Tasks

1. **Create pattern detection interfaces and data structures** (20 min)
   - IFusionPatternDetector
   - FusionCandidate record
   - FusionPatternType enum

2. **Implement graph analyzer** (25 min)
   - DependencyGraph construction
   - Linear chain detection
   - Memory access pattern analysis

3. **Implement element-wise pattern detector** (20 min)
   - Detect sequential element-wise ops
   - Validate compatibility
   - Score fusion benefit

4. **Implement conv-activation pattern detector** (20 min)
   - Detect conv + activation pairs
   - Validate parameter compatibility
   - Support standard activations (ReLU, Sigmoid, Tanh)

5. **Implement conv-batchnorm folding detector** (25 min)
   - Detect conv + BN patterns
   - Validate BN for inference folding
   - Handle different conv types (1D, 2D, 3D)

6. **Implement operation compatibility checker** (30 min)
   - Memory layout compatibility
   - Numerical precision checks
   - Thread block configuration
   - Side effect detection

## Test Cases

```csharp
[Test]
public void DetectElementWiseChain_ReturnsCandidate()
{
    var graph = BuildGraph(new[] { "Add", "Mul", "ReLU" });
    var detector = new ElementWisePatternDetector();
    var candidates = detector.DetectPatterns(graph);

    Assert.AreEqual(1, candidates.Count);
    Assert.AreEqual(FusionPatternType.ElementWise, candidates[0].PatternType);
}

[Test]
public void DetectConvReLU_ReturnsCandidate()
{
    var graph = BuildGraph(new[] { "Conv2D", "ReLU" });
    var detector = new ConvActivationPatternDetector();
    var candidates = detector.DetectPatterns(graph);

    Assert.AreEqual(1, candidates.Count);
    Assert.AreEqual(2, candidates[0].Operations.Count);
}

[Test]
public void IncompatibleLayouts_CannotFuse()
{
    var op1 = CreateOpWithLayout(TensorLayout.NCHW);
    var op2 = CreateOpWithLayout(TensorLayout.NHWC);
    var checker = new OperationCompatibilityChecker();

    Assert.IsFalse(checker.CanFuse(op1, op2));
}
```

## Success Criteria
- Pattern detectors correctly identify fusible sequences
- Compatibility checks prevent invalid fusions
- Benefit scoring prioritizes high-impact patterns
- No false positives in pattern detection

## Dependencies
- ComputationalGraph IR
- Operation abstraction
- Tensor data structures
