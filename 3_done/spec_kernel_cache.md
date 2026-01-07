# Spec: Kernel Cache for Dynamic Shapes

## Overview
Implement caching system for compiled kernels indexed by shape signatures to avoid recompilation.

## Requirements

### Struct: ShapeSignature
- Properties:
  - `OperationName`: string
  - `InputShapes`: int[][] - List of concrete shape arrays
  - `Hash`: int - Precomputed hash for fast lookup

- Methods:
  - `Equals(ShapeSignature other)`: bool
  - `GetHashCode()`: int
  - `ToString()`: string

- Static factory:
  - `Create(string opName, List<int[]> shapes)`: ShapeSignature

### Class: KernelCacheEntry
- Properties:
  - `Signature`: ShapeSignature
  - `CompiledKernel`: IntPtr / object (depends on backend)
  - `LastUsed`: DateTime
  - `UseCount`: int
  - `CompilationTimeMs`: long

- Methods:
  - `UpdateAccessTime()`: void
  - `IncrementUseCount()`: void

### Interface: IKernelCache
- Methods:
  - `Get(ShapeSignature sig)`: TKernel?
  - `Set(ShapeSignature sig, TKernel kernel)`: void
  - `Contains(ShapeSignature sig)`: bool
  - `Remove(ShapeSignature sig)`: void
  - `Clear()`: void
  - `GetStats()`: CacheStats

### Class: LRUKernelCache : IKernelCache
- Properties:
  - `MaxSize`: int - Maximum number of kernels to cache
  - `CurrentSize`: int

- Methods:
  - Constructor(int maxSize)
  - Eviction policy: LRU (Least Recently Used)
  - `EvictLeastRecentlyUsed()`: void
  - `GetEvictionCandidates(int count)`: List<ShapeSignature>

### Class: CacheStats
- Properties:
  - `TotalKernels`: int
  - `TotalHits`: long
  - `TotalMisses`: long
  - `HitRate`: double
  - `TotalCompilationTimeMs`: long
  - `AverageCompilationTimeMs`: double

- Methods:
  - `Reset()`: void

### Unit Tests
- Test ShapeSignature hashing and equality
- Test cache get/set operations
- Test LRU eviction policy
- Test stats tracking
- Test thread-safety (if applicable)
- Test cache limits and eviction

## Implementation Notes
- Use ConcurrentDictionary for thread-safe access
- Hash shape arrays efficiently (combine all dims into single hash)
- Consider partial matching for symbolic shapes (future enhancement)
- Log cache hits/misses for debugging
- Support persisting cache to disk (optional)

## Dependencies
- spec_symbolic_shape.md

## Success Criteria
- Fast lookup via hash-based indexing
- LRU eviction keeps hot kernels in cache
- Stats provide insights into cache effectiveness
- Thread-safe for concurrent compilation
- Memory bounded by max size
