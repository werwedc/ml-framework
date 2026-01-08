# Spec: Lazy Kernel Compilation

## Overview
Implement lazy compilation strategy where kernels are compiled on first use with actual shapes, then cached.

## Requirements

### Interface: IKernelCompiler
- Methods:
  - `Compile(Operation op, List<int[]> inputShapes, List<int[]> outputShapes)`: CompiledKernel
  - `CanCompile(Operation op)`: bool

### Interface: IKernelExecutor
- Methods:
  - `Execute(CompiledKernel kernel, List<Tensor> inputs, List<Tensor> outputs)`: void

### Class: LazyCompilationContext
- Properties:
  - `Operation`: Operation
  - `InputShapes`: List<int[]>
  - `OutputShapes`: List<int[]>
  - `IsCompiled`: bool
  - `CompiledKernel`: CompiledKernel?
  - `CompilationTimeMs`: long

- Methods:
  - `EnsureCompiled(IKernelCompiler compiler)`: void
  - `Execute(IKernelExecutor executor, List<Tensor> inputs, List<Tensor> outputs)`: void

### Class: LazyCompilationManager
- Properties:
  - `Cache`: IKernelCache<CompiledKernel>
  - `Compiler`: IKernelCompiler

- Methods:
  - `GetOrCompile(Operation op, List<int[]> inputShapes)`: CompiledKernel
  - `Precompile(Operation op, List<List<int[]>> shapeVariants)`: void - Warm up cache
  - `ClearCache()`: void
  - `GetCompilationStats()`: CompilationStats

### Class: CompilationStats
- Properties:
  - `TotalCompilations`: int
  - `CacheHits`: int
  - `CacheMisses`: int
  - `TotalCompilationTimeMs`: long
  - `UniqueKernels`: int

- Methods:
  - `ToReport()`: string

### Class: ShapeVariantGenerator
- Methods:
  - `GenerateVariants(SymbolicShape shape, int count)`: List<int[]> - Generate concrete shape variants
  - `GenerateGrid(List<SymbolicShape> shapes, List<int> samplesPerDim)`: List<List<int[]>>

### Unit Tests
- Test lazy compilation triggers on first use
- Test cache hit on subsequent uses
- Test precompilation with shape variants
- Test stats tracking
- Test shape variant generation
- Test concurrent compilation (if applicable)

## Implementation Notes
- Use thread-safe compilation (avoid compiling same kernel twice)
- Provide async compilation option for background warm-up
- Log compilation events for debugging
- Support compilation timeout
- Consider shape profiling for intelligent precompilation

## Dependencies
- spec_kernel_cache.md
- spec_symbolic_shape.md

## Success Criteria
- Kernels compiled only when needed
- Subsequent executions use cached kernels
- Precompilation warms up cache effectively
- Stats provide visibility into compilation behavior
- No duplicate compilations for same shapes
