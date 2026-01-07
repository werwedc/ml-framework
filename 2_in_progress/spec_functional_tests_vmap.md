# Spec: Tests for Vectorization (vmap)

## Overview
Create comprehensive unit tests for the vmap transformation to ensure it correctly vectorizes functions for batch processing.

## Scope
- Test single-input vectorization
- Test double-input vectorization
- Test multi-axis vectorization
- Test edge cases and error handling

## Test Files to Create
1. `tests/MLFramework.Tests/Functional/VMapTests.cs`

## Test Requirements

### 1. Single-Input VMap Tests

```csharp
using Xunit;
using MLFramework.Functional;

namespace MLFramework.Tests.Functional
{
    public class VMapTests
    {
        [Fact]
        public void Vectorize_SingleInput_ShouldProcessBatch()
        {
            // Arrange
            Func<Tensor, Tensor> addOne = t => t + Tensor.Ones(t.Shape);
            var batchedAddOne = Functional.Vectorize(addOne, axis: 0);

            // Act
            var input = Tensor.FromArray(new[] { 1f, 2f, 3f }).Reshape(3, 1);  // Shape [3, 1]
            var result = batchedAddOne(input);

            // Assert
            Assert.Equal(new[] { 3, 1 }, result.Shape);
            // Each element should be incremented by 1
            Assert.Equal(2f, result[0, 0].ToScalar());
            Assert.Equal(3f, result[1, 0].ToScalar());
            Assert.Equal(4f, result[2, 0].ToScalar());
        }

        [Fact]
        public void Vectorize_SingleInput_DifferentAxis()
        {
            // Arrange
            Func<Tensor, Tensor> multiplyByTwo = t => t * 2f;
            var batchedMultiply = Functional.Vectorize(multiplyByTwo, axis: 1);

            // Act
            var input = Tensor.FromArray(new[] { 1f, 2f, 3f, 4f }).Reshape(2, 2);  // Shape [2, 2]
            var result = batchedMultiply(input);

            // Assert
            Assert.Equal(new[] { 2, 2 }, result.Shape);
            // Each element should be multiplied by 2
            Assert.Equal(2f, result[0, 0].ToScalar());
            Assert.Equal(4f, result[0, 1].ToScalar());
            Assert.Equal(6f, result[1, 0].ToScalar());
            Assert.Equal(8f, result[1, 1].ToScalar());
        }

        [Fact]
        public void Vectorize_SingleInput_3DTensor()
        {
            // Arrange
            Func<Tensor, Tensor> identity = t => t;
            var batchedIdentity = Functional.Vectorize(identity, axis: 0);

            // Act
            var input = Tensor.FromArray(new float[24]).Reshape(4, 3, 2);  // Shape [4, 3, 2]
            var result = batchedIdentity(input);

            // Assert
            Assert.Equal(new[] { 4, 3, 2 }, result.Shape);
        }
    }
}
```

### 2. Double-Input VMap Tests

```csharp
public class VMapTests
{
    [Fact]
    public void Vectorize_DoubleInput_ShouldProcessBothBatches()
    {
        // Arrange
        Func<Tensor, Tensor, Tensor> add = (a, b) => a + b;
        var batchedAdd = Functional.Vectorize(add, axis: 0);

        // Act
        var input1 = Tensor.FromArray(new[] { 1f, 2f, 3f }).Reshape(3, 1);
        var input2 = Tensor.FromArray(new[] { 10f, 20f, 30f }).Reshape(3, 1);
        var result = batchedAdd(input1, input2);

        // Assert
        Assert.Equal(new[] { 3, 1 }, result.Shape);
        Assert.Equal(11f, result[0, 0].ToScalar());
        Assert.Equal(22f, result[1, 0].ToScalar());
        Assert.Equal(33f, result[2, 0].ToScalar());
    }

    [Fact]
    public void Vectorize_DoubleInput_WithDifferentShapes()
    {
        // Arrange
        Func<Tensor, Tensor, Tensor> matmul = (a, b) => a.MatMul(b);
        var batchedMatMul = Functional.Vectorize(matmul, axis: 0);

        // Act
        var input1 = Tensor.FromArray(new float[6]).Reshape(2, 3, 4);  // Batch of [3, 4]
        var input2 = Tensor.FromArray(new float[8]).Reshape(2, 4, 5);  // Batch of [4, 5]
        var result = batchedMatMul(input1, input2);

        // Assert
        Assert.Equal(new[] { 2, 3, 5 }, result.Shape);
    }
}
```

### 3. Multi-Axis VMap Tests

```csharp
public class VMapTests
{
    [Fact]
    public void Vectorize_WithInAxes_NullAxis_ShouldNotVectorize()
    {
        // Arrange
        Func<Tensor, Tensor, Tensor> multiply = (a, b) => a * b;
        var batchedMultiply = Functional.Vectorize(multiply, new object[] { 0, null });

        // Act
        var batchedInput = Tensor.FromArray(new[] { 1f, 2f }).Reshape(2, 1);  // [2, 1]
        var singleInput = Tensor.FromArray(new[] { 10f }).Reshape(1);  // [1]
        var result = batchedMultiply(batchedInput, singleInput);

        // Assert
        Assert.Equal(new[] { 2, 1 }, result.Shape);
        // Each batch element should be multiplied by the same single input
        Assert.Equal(10f, result[0, 0].ToScalar());
        Assert.Equal(20f, result[1, 0].ToScalar());
    }

    [Fact]
    public void Vectorize_WithInAxes_DifferentAxes()
    {
        // Arrange
        Func<Tensor, Tensor, Tensor> add = (a, b) => a + b;

        // Act
        var batchedAdd = Functional.Vectorize(add, new object[] { 0, 1 });

        var input1 = Tensor.FromArray(new float[6]).Reshape(2, 3);  // [2, 3]
        var input2 = Tensor.FromArray(new float[6]).Reshape(2, 3);  // [2, 3]
        var result = batchedAdd(input1, input2);

        // Assert
        Assert.Equal(new[] { 2, 3 }, result.Shape);
        // Should add corresponding elements
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.Equal(input1[i, j].ToScalar() + input2[i, j].ToScalar(),
                             result[i, j].ToScalar());
            }
        }
    }

    [Fact]
    public void Vectorize_WithInAxes_AllNull_ShouldThrow()
    {
        // Arrange
        Func<Tensor, Tensor, Tensor> add = (a, b) => a + b;

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            Functional.Vectorize(add, new object[] { null, null }));
    }
}
```

### 4. Error Handling Tests

```csharp
public class VMapTests
{
    [Fact]
    public void Vectorize_ShouldThrowForInvalidAxis()
    {
        // Arrange
        Func<Tensor, Tensor> identity = t => t;
        var input = Tensor.FromArray(new[] { 1f, 2f }).Reshape(2, 1);

        // Act & Assert
        var batchedIdentity = Functional.Vectorize(identity, axis: 5);
        Assert.Throws<ArgumentException>(() => batchedIdentity(input));
    }

    [Fact]
    public void Vectorize_DoubleInput_ShouldThrowForMismatchedBatchSizes()
    {
        // Arrange
        Func<Tensor, Tensor, Tensor> add = (a, b) => a + b;
        var batchedAdd = Functional.Vectorize(add, axis: 0);

        var input1 = Tensor.FromArray(new[] { 1f, 2f }).Reshape(2, 1);  // Batch size 2
        var input2 = Tensor.FromArray(new[] { 10f, 20f, 30f }).Reshape(3, 1);  // Batch size 3

        // Act & Assert
        Assert.Throws<ArgumentException>(() => batchedAdd(input1, input2));
    }

    [Fact]
    public void Vectorize_WithInAxes_ShouldThrowForWrongLength()
    {
        // Arrange
        Func<Tensor, Tensor, Tensor> add = (a, b) => a + b;

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            Functional.Vectorize(add, new object[] { 0 }));  // Should have 2 axes
    }

    [Fact]
    public void Vectorize_ShouldThrowForNonTensorReturn()
    {
        // Arrange
        Func<Tensor, int> getLength = t => t.Shape.TotalElements;

        // Act & Assert
        Assert.Throws<NotSupportedException>(() =>
            Functional.Vectorize(getLength));
    }
}
```

### 5. Real-World Scenario Tests

```csharp
public class VMapTests
{
    [Fact]
    public void Vectorize_ComputeLossForBatch()
    {
        // Arrange
        Func<Tensor, Tensor, Tensor> mseLoss = (predictions, targets) =>
        {
            var diff = predictions - targets;
            return (diff * diff).Mean();
        };

        var batchedMSE = Functional.Vectorize(mseLoss, axis: 0);

        // Act
        var predictions = Tensor.FromArray(new[] { 1f, 2, 3, 4 }).Reshape(4, 1);
        var targets = Tensor.FromArray(new[] { 1.1f, 2.2f, 2.9f, 4.1f }).Reshape(4, 1);
        var result = batchedMSE(predictions, targets);

        // Assert
        Assert.Single(result.Shape);  // Scalar result
        Assert.True(result.ToScalar() > 0);
    }

    [Fact]
    public void Vectorize_SoftmaxBatch()
    {
        // Arrange
        Func<Tensor, Tensor> softmax = t => t.Exp() / t.Exp().Sum();

        var batchedSoftmax = Functional.Vectorize(softmax, axis: 0);

        // Act
        var input = Tensor.FromArray(new[] { 1f, 2, 3, 1, 2, 3 }).Reshape(2, 3);
        var result = batchedSoftmax(input);

        // Assert
        Assert.Equal(new[] { 2, 3 }, result.Shape);

        // Each row should sum to 1 (approximately)
        for (int i = 0; i < 2; i++)
        {
            float rowSum = 0;
            for (int j = 0; j < 3; j++)
                rowSum += result[i, j].ToScalar();
            Assert.InRange(rowSum, 0.99f, 1.01f);
        }
    }
}
```

## Files to Create
1. `tests/MLFramework.Tests/Functional/VMapTests.cs`

## Dependencies
- spec_functional_vmap_basic.md (implementation must be complete)
- spec_functional_vmap_multi_axis.md (implementation must be complete)

## Success Criteria
- All tests pass
- Tests cover single-input, double-input, and multi-axis scenarios
- Error cases are properly tested
- Real-world use cases are tested
- Tests are maintainable and well-documented

## Notes for Coder
- Use xUnit for testing
- Create helper methods for creating test tensors
- Use Assert.InRange for floating-point comparisons
- Test edge cases: empty batches, single-element batches, large batches
- Verify that the function semantics are preserved (no unexpected side effects)
- Consider performance for larger batches (add timing tests if needed)

## Additional Considerations
- Test that the original function is not modified by vectorization
- Test that vectorization is idempotent (applying vmap twice should handle correctly)
- Test with tensors that have different dtypes (float32, float64)
- Consider adding integration tests with actual ML operations (matmul, conv, etc.)
