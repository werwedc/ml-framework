# Spec: FunctionContext Implementation

## Overview
Implement the `FunctionContext` class that provides the mechanism to save and retrieve state (tensors and objects) between forward and backward passes in custom autograd functions.

## Requirements

### 1. Class Definition
Create a `FunctionContext` class in `src/Autograd/FunctionContext.cs` with the following responsibilities:
- Store tensors and objects during the forward pass
- Retrieve saved state during the backward pass
- Automatically clean up saved state after backward pass
- Support multiple independent contexts for functions called multiple times

### 2. Core Methods

#### SaveForBackward(params Tensor[] tensors)
- Accepts 0 or more Tensor parameters
- Stores tensors in an internal list for later retrieval
- Preserves reference to the actual tensor objects
- Validates that tensors are not null

#### SaveForBackward(params object[] objects)
- Accepts 0 or more generic objects
- Supports saving scalars, integers, strings, or any other metadata
- Stores in a separate list from tensors for type safety

#### Tensor GetSavedTensor(int index)
- Retrieves a tensor at the specified index from the tensor list
- Validates index is within bounds
- Throws `ArgumentOutOfRangeException` with clear message if index is invalid
- Returns null if tensor at index is null (but should still throw if index out of bounds)

#### object GetSavedObject(int index)
- Retrieves an object at the specified index from the object list
- Validates index is within bounds
- Throws `ArgumentOutOfRangeException` with clear message if index is invalid
- Returns null if object at index is null (but should still throw if index out of bounds)

### 3. Memory Management
- Implement `Dispose()` method to clear all saved state
- Implement `Clear()` method to release references to tensors and objects
- Provide a boolean property `IsDisposed` to track cleanup status

### 4. Properties
- `SavedTensorCount` - Returns the number of saved tensors (read-only)
- `SavedObjectCount` - Returns the number of saved objects (read-only)
- `IsDisposed` - Returns whether the context has been disposed (read-only)

### 5. Error Handling
All methods should provide clear error messages:
- "Cannot retrieve tensor from disposed context"
- "Tensor index {index} is out of range. Valid range is 0 to {count}"
- "Object index {index} is out of range. Valid range is 0 to {count}"

## Implementation Notes

### Internal Data Structure
```csharp
private List<Tensor> _savedTensors = new List<Tensor>();
private List<object> _savedObjects = new List<object>();
private bool _disposed = false;
```

### Example Usage
```csharp
var ctx = new FunctionContext();

// Save state in forward
ctx.SaveForBackward(tensor1, tensor2);
ctx.SaveForBackward(scalarValue, "metadata");

// Retrieve in backward
var t1 = ctx.GetSavedTensor(0);
var t2 = ctx.GetSavedTensor(1);
var scalar = ctx.GetSavedObject(0);

// Cleanup
ctx.Clear();  // or ctx.Dispose()
```

## Testing Requirements
Create unit tests in `tests/Autograd/FunctionContextTests.cs`:

1. **Basic Save/Retrieve Tests**
   - Save multiple tensors and retrieve them by index
   - Save multiple objects and retrieve them by index
   - Save both tensors and objects independently

2. **Edge Cases**
   - Save empty arrays (should work)
   - Retrieve from empty context (should throw)
   - Save null values (should be stored and retrievable as null)

3. **Memory Management Tests**
   - Verify Clear() empties the lists
   - Verify Dispose() works and sets IsDisposed flag
   - Verify retrieval throws after disposal

4. **Error Message Tests**
   - Verify ArgumentOutOfRangeException messages for invalid indices
   - Verify appropriate error for operations on disposed context

## Success Criteria
- [ ] FunctionContext class is implemented in `src/Autograd/FunctionContext.cs`
- [ ] All four core methods (SaveForBackward x2, GetSavedTensor, GetSavedObject) work correctly
- [ ] Memory cleanup via Clear() and Dispose() is implemented
- [ ] All error messages are clear and informative
- [ ] Unit tests cover all scenarios with >90% code coverage
