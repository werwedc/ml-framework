# Spec: Tensor Gradient Tracking Infrastructure

## Overview
Implement the core gradient tracking infrastructure for Tensor class, enabling automatic gradient recording and storage during forward propagation.

## Files to Create
- `src/MLFramework/Autograd/ITensorGrad.cs`
- `src/MLFramework/Autograd/TensorGradStorage.cs`
- `tests/MLFramework.Tests/Autograd/TensorGradTests.cs`

## API Design

### Interface: ITensorGrad
```csharp
public interface ITensorGrad
{
    bool RequiresGrad { get; }
    Tensor Grad { get; set; }
    bool IsLeaf { get; }
    void ZeroGrad();
    void DetachGrad();
    void AccumulateGrad(Tensor gradient);
}
```

### Class: TensorGradStorage
```csharp
public class TensorGradStorage
{
    public bool RequiresGrad { get; private set; }
    public Tensor Grad { get; private set; }
    public bool IsLeaf { get; private set; }

    public TensorGradStorage(bool requiresGrad, bool isLeaf = true);
    public void ZeroGrad();
    public void DetachGrad();
    public void AccumulateGrad(Tensor gradient);
    public void SetGrad(Tensor gradient);
}
```

## Requirements

### Core Functionality
1. **Gradient Storage**
   - Store gradient tensor when `requiresGrad` is true
   - Null gradient for tensors that don't require gradients
   - Lazy initialization of gradient tensor

2. **Leaf vs Non-Leaf Tracking**
   - Leaf tensors: user-created or `.Detach()` tensors
   - Non-leaf tensors: intermediate results from operations
   - `IsLeaf` property indicates if tensor is a leaf

3. **Gradient Accumulation**
   - Support multiple backward passes without clearing gradients
   - `ZeroGrad()` method to clear accumulated gradients
   - `AccumulateGrad()` adds to existing gradient

4. **Gradient Detachment**
   - `DetachGrad()` removes from computation graph
   - Sets `requiresGrad` to false
   - Preserves tensor data

## Implementation Notes

### Memory Management
- Only allocate gradient tensor when needed
- Dispose gradients when tensor is disposed
- Consider weak references for intermediate tensors

### Thread Safety
- Gradient accumulation should be thread-safe
- Use lock-free atomic operations where possible
- Support for async gradient computation

## Testing Requirements

### Unit Tests
1. Create tensor with `requiresGrad: true` → verify gradient tracking enabled
2. Create tensor with `requiresGrad: false` → verify no gradient tracking
3. Call `ZeroGrad()` → verify gradient is null
4. Call `AccumulateGrad()` twice → verify gradients are summed
5. Call `DetachGrad()` → verify `requiresGrad` is false and gradient is null
6. Test leaf vs non-leaf tensor identification
7. Test gradient accumulation with multiple threads

## Dependencies
- Core Tensor class
- Memory management system

## Success Criteria
- Tensor can track gradient requirement state
- Gradient storage is memory-efficient
- Thread-safe gradient accumulation
- Clear separation between leaf and non-leaf tensors
