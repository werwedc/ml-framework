# Broadcasting Mechanism for Tensor Operations

## Overview
Implement NumPy-style broadcasting to allow element-wise operations between tensors of different shapes. This is a fundamental feature that dramatically increases flexibility and reduces the need for manual reshaping.

## Problem Statement
Currently, tensor operations require exact shape matches. This is restrictive and verbose. For example, adding a bias vector to a batch of data requires manual expansion:
```csharp
// Current limitation - shapes must match exactly
var batch = Tensor.Zeros(new int[] { 32, 100 });  // 32 samples, 100 features
var bias = Tensor.Zeros(new int[] { 32, 100 });   // Had to manually expand
var result = batch + bias;
```

Broadcasting would enable:
```csharp
var batch = Tensor.Zeros(new int[] { 32, 100 });  // 32 samples, 100 features
var bias = Tensor.Zeros(new int[] { 100 });       // Just the bias vector
var result = batch + bias;  // Automatically broadcasts to match batch shape
```

## Feature Requirements

### 1. Broadcasting Rules Implementation
- Implement NumPy-compatible broadcasting rules
- Align shapes on the right (trailing dimensions)
- Dimensions are compatible when:
  - They are equal, OR
  - One of them is 1, OR
  - One dimension doesn't exist (implicit 1)
- Result shape is element-wise maximum along each dimension

### 2. Binary Operations with Broadcasting
- Addition (`+`): Support broadcasting
- Subtraction (`-`): Support broadcasting
- Element-wise multiplication (`*`): Support broadcasting (scalar already works)
- Element-wise division (`/`): New operator with broadcasting
- Element-wise power (`^`): New operator with broadcasting

### 3. Broadcasting Validation
- Throw descriptive exceptions when shapes are incompatible
- Error messages should show: input shapes, expected broadcast, and why it failed

### 4. Explicit Broadcasting Operations
- `Tensor.ExpandTo(Tensor reference)` - manually expand tensor to match another's shape
- `Tensor.BroadcastTo(int[] targetShape)` - expand to specific shape
- `bool CanBroadcastTo(int[] targetShape)` - check if broadcasting is possible

## Use Cases

### Neural Network Bias Addition
```csharp
// Forward pass with bias
var activations = Tensor.Zeros(new int[] { batch, hiddenSize });
var bias = Tensor.Zeros(new int[] { hiddenSize });
var output = activations + bias;  // Broadcasting works!
```

### Batch Normalization
```csharp
// Normalize per-feature across batch
var batch = Tensor.Random(new int[] { 32, 100 });
var mean = batch.Mean(new int[] { 0 });  // Shape: [100]
var variance = batch.Variance(new int[] { 0 });  // Shape: [100]
var normalized = (batch - mean) / Tensor.Sqrt(variance + 1e-5f);
```

### Element-wise Operations with Scalars and Vectors
```csharp
var data = Tensor.Random(new int[] { 10, 20, 30 });
var weights = Tensor.Random(new int[] { 20 });  // Broadcasts across first and third dims
var weighted = data * weights;  // Shape: [10, 20, 30]
```

## Technical Implementation Considerations

### Shape Resolution
```csharp
private static (int[] shape, int[][] strides) ResolveBroadcastShape(int[] shapeA, int[] shapeB)
{
    // Determine max dimensions
    // Align from right
    // Check compatibility
    // Compute resulting shape
    // Compute broadcast strides for each input
}
```

### Memory Strategy
- **Option A**: Allocate new arrays for broadcast results (simpler, more memory)
- **Option B**: Use virtual broadcasting with stride manipulation (efficient, less memory)
- **Recommendation**: Start with Option A, optimize later with stride tricks

### Gradient Computation
- Backward pass must handle broadcasting correctly
- Gradients are summed along broadcasted dimensions
- Example: If `A + B` broadcasted B, `B.grad = sum(A.grad)` along broadcasted axes

## Benefits
1. **Expressiveness**: Write more concise, readable code
2. **Performance**: Avoid unnecessary memory allocations and copies
3. **Compatibility**: Match API expectations from PyTorch/NumPy users
4. **Flexibility**: Support common ML patterns (bias, normalization, scaling)

## Priority: HIGH
This is a foundational feature that enables many advanced operations and is expected in any tensor library.

## Dependencies
- Existing Tensor infrastructure
- Gradient computation system
- Tensor indexing and shape management
