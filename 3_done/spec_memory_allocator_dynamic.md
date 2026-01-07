# Spec: Memory Allocator for Dynamic Shapes

## Overview
Implement memory allocation strategy that handles variable-size tensors efficiently using predictive allocation.

## Requirements

### Class: ShapeBounds
- Properties:
  - `MinShape`: int[]
  - `MaxShape`: int[]
  - `ExpectedShape`: int[]

- Methods:
  - `CalculateMaxElements()`: long
  - `CalculateExpectedElements()`: long
  - `Contains(int[] shape)`: bool

### Interface: IDynamicMemoryAllocator
- Methods:
  - `Allocate(ShapeBounds bounds)`: IMemoryHandle
  - `Resize(IMemoryHandle handle, int[] newShape)`: void
  - `Free(IMemoryHandle handle)`: void
  - `GetAllocationStats()`: AllocationStats

### Class: PredictiveAllocator : IDynamicMemoryAllocator
- Properties:
  - `PaddingFactor`: double - Allocate X% more than expected (default 1.2)
  - `MaxCapacity`: long bytes

- Methods:
  - Constructor(IPaddingStrategy paddingStrategy)
  - Allocate based on expected shape with padding
  - `UpdateExpectations(IMemoryHandle handle, int[] actualShape)`: void - Learn actual usage

### Interface: IPaddingStrategy
- Methods:
  - `CalculateRequiredSize(ShapeBounds bounds)`: long
  - `ShouldResize(IMemoryHandle handle, int[] newShape)`: bool

### Class: AdaptivePaddingStrategy : IPaddingStrategy
- Properties:
  - `GrowthFactor`: double (default 1.5)
  - `ShrinkThreshold`: double (default 0.5) - Shrink if usage < threshold

- Methods:
  - Grow allocations if new shape exceeds capacity
  - Shrink allocations if usage consistently low
  - Track actual usage patterns

### Class: MemoryHandle : IMemoryHandle
- Properties:
  - `Pointer`: IntPtr
  - `CapacityBytes`: long
  - `CurrentShape`: int[]
  - `ShapeBounds`: ShapeBounds
  - `AllocationTime`: DateTime

- Methods:
  - `Resize(int[] newShape)`: void
  - `GetEffectiveSize()`: long - Calculate actual size for current shape
  - `GetUtilization()`: double - Current size / capacity

### Class: AllocationStats
- Properties:
  - `TotalAllocations`: int
  - `TotalResizes`: int
  - `TotalBytesAllocated`: long
  - `TotalBytesWasted`: long - Padding overhead
  - `AverageUtilization`: double

- Methods:
  - `ToReport()`: string

### Unit Tests
- Test allocation with various shape bounds
- Test resizing logic
- Test padding strategies
- Test utilization calculation
- Test stats tracking
- Test capacity limits

## Implementation Notes
- Use arena allocation for efficiency when possible
- Reuse freed buffers for similar allocations
- Track per-tensor usage patterns for better predictions
- Support pinned memory for GPU transfers
- Log allocation events for debugging

## Dependencies
- spec_symbolic_shape.md

## Success Criteria
- Allocates sufficient memory for shape variations
- Minimizes waste through adaptive padding
- Supports efficient resizing without reallocation when possible
- Utilization stats inform padding decisions
- Memory bounded by global limits
