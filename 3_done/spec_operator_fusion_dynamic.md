# Spec: Operator Fusion with Dynamic Shapes

## Overview
Implement operator fusion that preserves shape dynamics to enable optimizations for variable-size tensors.

## Requirements

### Class: FusionNode
- Properties:
  - `Operations`: List<Operation>
  - `InputShapes`: List<SymbolicShape>
  - `OutputShapes`: List<SymbolicShape>
  - `FusionId`: string

- Methods:
  - `CanFuseWith(Operation nextOp, List<SymbolicShape> intermediateShapes)`: bool
  - `AddOperation(Operation op)`: void
  - `GetFusedSignature()`: string
  - `ValidateFusion()`: bool

### Class: FusionCandidateAnalyzer
- Methods:
  - `FindFusibleOperations(List<Operation> ops)`: List<FusionNode>
  - `AnalyzeBenefit(FusionNode node)`: FusionBenefit
  - `IsShapePreserving(Operation op)`: bool
  - `RequiresRuntimeShapeCheck(FusionNode node)`: bool

### Class: FusionBenefit
- Properties:
  - `EstimatedSpeedup`: double
  - `MemorySaved`: long bytes
  - `KernelCountReduction`: int
  - `ComplexityScore`: double

- Methods:
  - `ShouldFuse(double threshold)`: bool

### Dynamic Fusion Rules:

1. **Element-wise Fusion**: Fuse sequence of element-wise operations
   - Shape must be preserved or transformed predictably
   - Support broadcasting within fusion

2. **MatMul + Activation**: Fuse matmul with following activation
   - Output shape must be [M, N] for both
   - Handle symbolic M, N dimensions

3. **Conv + BatchNorm + Activation**: Fuse CNN layers
   - Spatial dimensions must be symbolic but consistent
   - Validate output shapes match expectations

4. **Reduce + Element-wise**: Fuse reduction with following ops
   - Reduced dimension must be tracked symbolically

### Class: RuntimeShapeInjector
- Methods:
  - `InjectShapeCheck(FusionNode node)`: List<Operation> - Add runtime shape validation
  - `GenerateShapeDispatch(FusionNode node)`: Operation - Dispatch to specialized kernel based on shape
  - `GenerateGenericFallback(FusionNode node)`: Operation - Generic kernel for unknown shapes

### Class: FusionKernelGenerator
- Methods:
  - `GenerateFusedKernel(FusionNode node, List<int[]> concreteShapes)`: CompiledKernel
  - `GenerateGenericKernel(FusionNode node)`: CompiledKernel
  - `CanGenerateSpecialized(List<int[]> shapes)`: bool

### Unit Tests
- Test fusion node creation and validation
- Test fusion candidate analysis
- Test benefit estimation
- Test shape-preserving operations
- Test dynamic fusion rules
- Test runtime shape injection
- Test kernel generation for fused ops
- Test generic vs specialized kernel selection

## Implementation Notes
- Fusion must preserve shape semantics
- Use symbolic shapes to ensure fusion is valid across shape variations
- Generate both specialized (for common shapes) and generic kernels
- Runtime shape checks ensure correct kernel selection
- Cache fused kernels by shape signature

## Dependencies
- spec_symbolic_shape.md
- spec_shape_inference_engine.md
- spec_kernel_cache.md

## Success Criteria
- Identifies fusible operations with dynamic shapes
- Fusion preserves shape semantics
- Generates efficient specialized kernels
- Provides generic fallback for uncommon shapes
- Runtime dispatch selects correct kernel
