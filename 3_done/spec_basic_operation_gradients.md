# Spec: Basic Operation Gradients

## Overview
Implement gradient computation functions for basic tensor operations (addition, subtraction, multiplication, division, element-wise operations, etc.).

## Files to Create
- `src/MLFramework/Autograd/Operations/AddGrad.cs`
- `src/MLFramework/Autograd/Operations/MulGrad.cs`
- `src/MLFramework/Autograd/Operations/SubGrad.cs`
- `src/MLFramework/Autograd/Operations/DivGrad.cs`
- `src/MLFramework/Autograd/Operations/PowGrad.cs`
- `src/MLFramework/Autograd/Operations/ExpGrad.cs`
- `src/MLFramework/Autograd/Operations/LogGrad.cs`
- `src/MLFramework/Autograd/Operations/SqrtGrad.cs`
- `src/MLFramework/Autograd/Operations/ReluGrad.cs`
- `src/MLFramework/Autograd/Operations/SigmoidGrad.cs`
- `src/MLFramework/Autograd/Operations/TanhGrad.cs`
- `src/MLFramework/Autograd/Operations/SumGrad.cs`
- `src/MLFramework/Autograd/Operations/MeanGrad.cs`
- `tests/MLFramework.Tests/Autograd/Operations/OperationGradTests.cs`

## API Design

### Interface: IOperationGrad
```csharp
public interface IOperationGrad
{
    string OperationName { get; }
    Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context);
}
```

### Example: AddGrad
```csharp
public class AddGrad : IOperationGrad
{
    public string OperationName => "Add";

    public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
    {
        // d(x + y)/dx = 1, d(x + y)/dy = 1
        return new Tensor[] { gradOutput.Clone(), gradOutput.Clone() };
    }
}
```

### Example: MulGrad
```csharp
public class MulGrad : IOperationGrad
{
    public string OperationName => "Mul";

    public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
    {
        // d(x * y)/dx = y, d(x * y)/dy = x
        Tensor y = inputs[1];
        Tensor x = inputs[0];
        return new Tensor[] { gradOutput * y, gradOutput * x };
    }
}
```

### Example: ReluGrad
```csharp
public class ReluGrad : IOperationGrad
{
    public string OperationName => "Relu";

    public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
    {
        // d(relu(x))/dx = 1 if x > 0 else 0
        Tensor x = inputs[0];
        Tensor mask = x.GreaterThan(0.0);
        return new Tensor[] { gradOutput * mask };
    }
}
```

## Requirements

### Binary Operations
1. **Addition/Subtraction**
   - Add: grad passes through unchanged
   - Sub: grad passes through (with sign flip for second input)
   - Handle broadcasting

2. **Multiplication/Division**
   - Mul: gradient involves other operand
   - Div: gradient involves numerator and denominator
   - Handle division by zero

3. **Power Operations**
   - Pow: d(x^n)/dx = n * x^(n-1)
   - Sqrt: special case of pow
   - Handle negative inputs

### Unary Operations
1. **Activation Functions**
   - ReLU: gradient is 0 or 1
   - Sigmoid: gradient = sigmoid(x) * (1 - sigmoid(x))
   - Tanh: gradient = 1 - tanh²(x)

2. **Math Functions**
   - Exp: gradient = exp(x)
   - Log: gradient = 1/x
   - Sqrt: gradient = 1/(2*sqrt(x))

### Reduction Operations
1. **Sum/Mean**
   - Sum: gradient = 1 for each element
   - Mean: gradient = 1/n for each element
   - Handle reduction axes

## Implementation Notes

### Broadcasting Support
- Gradient must broadcast back to original shape
- Use sum/reduce along broadcasted dimensions
- Preserve gradient shape for chain rule

### Numerical Stability
- Avoid division by zero in gradients
- Handle edge cases (e.g., log(0), sqrt(-1))
- Add small epsilon where needed

### Performance
- Reuse tensors where possible
- Avoid unnecessary allocations
- Vectorized operations for efficiency

### Integration with Tensor Operations
- Each operation registers its gradient function
- OperationContext saves inputs needed for backward
- Gradient functions use saved tensors from context

## Testing Requirements

### Unit Tests
1. Test Add gradient with same shapes
2. Test Add gradient with broadcasting
3. Test Mul gradient (verify chain rule)
4. Test Div gradient (verify chain rule)
5. Test Pow gradient (x^n, n constant)
6. Test Relu gradient (positive and negative inputs)
7. Test Sigmoid gradient
8. Test Tanh gradient
9. Test Exp gradient
10. Test Log gradient (handle x=0)
11. Test Sum gradient (verify broadcasts)
12. Test Mean gradient (verify 1/n scaling)

### Integration Tests
1. Build neural network layer (linear + activation) → verify gradients
2. Test operation chain (multiple operations) → verify gradient accuracy
3. Compare with numerical gradients for random tensors
4. Test gradients for complex expressions
5. Test gradient accuracy for edge cases

## Dependencies
- Tensor operations
- Computational graph infrastructure
- Backward pass implementation

## Success Criteria
- Accurate gradients for all basic operations
- Correct broadcasting in backward pass
- Handles edge cases (zeros, negative numbers)
- Efficient gradient computation
- Clear error messages for invalid operations
