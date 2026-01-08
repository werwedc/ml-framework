# Spec: Tests for Function Composition

## Overview
Create unit tests for the function composition utilities to ensure functional programming patterns work correctly.

## Scope
- Test Compose (right-to-left)
- Test Pipe (left-to-right)
- Test Partial application
- Test Currying
- Test utility functions (Identity, Constant, Tap)

## Test Files to Create
1. `tests/MLFramework.Tests/Functional/CompositionTests.cs`

## Test Requirements

### 1. Compose Tests

```csharp
using Xunit;
using MLFramework.Functional;

namespace MLFramework.Tests.Functional
{
    public class CompositionTests
    {
        [Fact]
        public void Compose_TwoFunctions_ShouldExecuteRightToLeft()
        {
            // Arrange
            Func<int, int> doubleIt = x => x * 2;
            Func<int, int> addFive = x => x + 5;

            // Act: Compose(addFive, doubleIt) means addFive(doubleIt(x))
            var composed = Functional.Compose(addFive, doubleIt);

            // Assert
            // composed(3) = addFive(doubleIt(3)) = addFive(6) = 11
            Assert.Equal(11, composed(3));
        }

        [Fact]
        public void Compose_ThreeFunctions_ShouldExecuteRightToLeft()
        {
            // Arrange
            Func<int, int> doubleIt = x => x * 2;
            Func<int, int> addFive = x => x + 5;
            Func<int, int> square = x => x * x;

            // Act: Compose(square, addFive, doubleIt) = square(addFive(doubleIt(x)))
            var composed = Functional.Compose(square, addFive, doubleIt);

            // Assert
            // composed(3) = square(addFive(doubleIt(3))) = square(addFive(6)) = square(11) = 121
            Assert.Equal(121, composed(3));
        }

        [Fact]
        public void Compose_ManyFunctions_ShouldExecuteCorrectly()
        {
            // Arrange
            Func<int, int>[] functions =
            {
                x => x + 1,
                x => x * 2,
                x => x + 3,
                x => x / 2
            };

            // Act
            var composed = Functional.Compose(functions);

            // Assert
            // ((x + 1) * 2 + 3) / 2
            // For x=5: ((5+1)*2+3)/2 = (6*2+3)/2 = (12+3)/2 = 15/2 = 7
            Assert.Equal(7, composed(5));
        }

        [Fact]
        public void Compose_TensorFunctions_ShouldWork()
        {
            // Arrange
            Func<Tensor, Tensor> doubleIt = t => t * 2f;
            Func<Tensor, Tensor> addOne = t => t + 1f;

            // Act
            var composed = Functional.Compose(addOne, doubleIt);

            var input = Tensor.FromArray(new[] { 1f, 2f, 3f });
            var result = composed(input);

            // Assert
            // result = addOne(doubleIt(input)) = (input * 2) + 1
            Assert.Equal(3f, result[0].ToScalar());  // (1*2)+1
            Assert.Equal(5f, result[1].ToScalar());  // (2*2)+1
            Assert.Equal(7f, result[2].ToScalar());  // (3*2)+1
        }

        [Fact]
        public void Compose_EmptyArray_ShouldThrow()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                Functional.Compose(Array.Empty<Func<int, int>>()));
        }

        [Fact]
        public void Compose_SingleFunction_ShouldReturnSameFunction()
        {
            // Arrange
            Func<int, int> doubleIt = x => x * 2;

            // Act
            var composed = Functional.Compose(doubleIt);

            // Assert
            Assert.Equal(6, composed(3));
        }
    }
}
```

### 2. Pipe Tests

```csharp
public class CompositionTests
{
    [Fact]
    public void Pipe_SingleFunction_ShouldApplyFunction()
    {
        // Arrange
        int value = 5;
        Func<int, int> doubleIt = x => x * 2;

        // Act
        var result = value.Pipe(doubleIt);

        // Assert
        Assert.Equal(10, result);
    }

    [Fact]
    public void Pipe_MultipleFunctions_ShouldApplyLeftToRight()
    {
        // Arrange
        int value = 5;
        Func<int, int>[] functions =
        {
            x => x + 1,      // 6
            x => x * 2,      // 12
            x => x - 3       // 9
        };

        // Act
        var result = value.Pipe(functions);

        // Assert
        Assert.Equal(9, result);
    }

    [Fact]
    public void Pipe_DifferentTypes_ShouldWork()
    {
        // Arrange
        int value = 5;

        // Act: int -> string -> int -> string
        var result = value.Pipe(
            (Func<int, string>)(x => x.ToString()),
            (Func<string, int>)(s => int.Parse(s) + 10),
            (Func<int, string>)(x => x.ToString()));

        // Assert
        Assert.Equal("15", result);
    }

    [Fact]
    public void Pipe_ChainingSyntax_ShouldWork()
    {
        // Arrange
        int value = 5;

        // Act
        var result = value
            .Pipe(x => x + 1)
            .Pipe(x => x * 2)
            .Pipe(x => x - 3);

        // Assert
        Assert.Equal(9, result);
    }

    [Fact]
    public void Pipe_EmptyArray_ShouldReturnValue()
    {
        // Arrange
        int value = 5;

        // Act
        var result = value.Pipe(Array.Empty<Func<int, int>>());

        // Assert
        Assert.Equal(5, result);
    }
}
```

### 3. Partial Application Tests

```csharp
public class CompositionTests
{
    [Fact]
    public void Partial_FirstArgument_ShouldBindValue()
    {
        // Arrange
        Func<int, int, int> add = (a, b) => a + b;

        // Act: Bind first argument to 5
        var addFive = Functional.Partial(add, 5);

        // Assert
        Assert.Equal(8, addFive(3));  // 5 + 3
        Assert.Equal(10, addFive(5));  // 5 + 5
    }

    [Fact]
    public void Partial_SecondArgument_ShouldBindValue()
    {
        // Arrange
        Func<int, int, int> multiply = (a, b) => a * b;

        // Act: Bind second argument to 10
        var multiplyByTen = Functional.PartialSecond(multiply, 10);

        // Assert
        Assert.Equal(20, multiplyByTen(2));   // 2 * 10
        Assert.Equal(50, multiplyByTen(5));   // 5 * 10
    }

    [Fact]
    public void Partial_ThreeArguments_ShouldBindFirst()
    {
        // Arrange
        Func<int, int, int, int> sumThree = (a, b, c) => a + b + c;

        // Act: Bind first argument to 5
        var addFive = Functional.Partial(sumThree, 5);

        // Assert
        Assert.Equal(10, addFive(3, 2));    // 5 + 3 + 2
        Assert.Equal(20, addFive(10, 5));   // 5 + 10 + 5
    }

    [Fact]
    public void Partial_TensorFunctions_ShouldWork()
    {
        // Arrange
        Func<Tensor, Tensor, Tensor> add = (a, b) => a + b;

        // Act
        var addOnes = Functional.Partial(add, Tensor.Ones(new[] { 3 }));

        var input = Tensor.FromArray(new[] { 1f, 2f, 3f });
        var result = addOnes(input);

        // Assert
        Assert.Equal(2f, result[0].ToScalar());
        Assert.Equal(3f, result[1].ToScalar());
        Assert.Equal(4f, result[2].ToScalar());
    }
}
```

### 4. Currying Tests

```csharp
public class CompositionTests
{
    [Fact]
    public void Curry_TwoArguments_ShouldTransform()
    {
        // Arrange
        Func<int, int, int> add = (a, b) => a + b;

        // Act: Curry add to get function that returns function
        var curried = Functional.Curry(add);

        // Assert
        var addFive = curried(5);
        Assert.Equal(8, addFive(3));  // 5 + 3
        Assert.Equal(15, addFive(10)); // 5 + 10
    }

    [Fact]
    public void Curry_TwoArguments_ShouldAllowChaining()
    {
        // Arrange
        Func<int, int, int> multiply = (a, b) => a * b;

        // Act
        var curried = Functional.Curry(multiply);

        // Assert
        var result = curried(2)(3);  // 2 * 3
        Assert.Equal(6, result);
    }

    [Fact]
    public void Curry_ThreeArguments_ShouldTransform()
    {
        // Arrange
        Func<int, int, int, int> sumThree = (a, b, c) => a + b + c;

        // Act
        var curried = Functional.Curry(sumThree);

        // Assert
        var addFive = curried(5);
        var addFiveAndThree = addFive(3);
        Assert.Equal(10, addFiveAndThree(2));  // 5 + 3 + 2
    }

    [Fact]
    public void Curry_ThreeArguments_ShouldAllowChaining()
    {
        // Arrange
        Func<int, int, int, int> multiply = (a, b, c) => a * b * c;

        // Act
        var curried = Functional.Curry(multiply);

        // Assert
        var result = curried(2)(3)(4);  // 2 * 3 * 4
        Assert.Equal(24, result);
    }

    [Fact]
    public void Uncurry_ShouldReverseCurrying()
    {
        // Arrange
        Func<int, int, int> originalAdd = (a, b) => a + b;
        var curried = Functional.Curry(originalAdd);

        // Act
        var uncurried = Functional.Uncurry(curried);

        // Assert
        Assert.Equal(8, uncurried(5, 3));
        Assert.Equal(15, uncurried(10, 5));
    }
}
```

### 5. Utility Function Tests

```csharp
public class CompositionTests
{
    [Fact]
    public void Identity_ShouldReturnInput()
    {
        // Arrange
        int value = 42;
        string text = "hello";

        // Act & Assert
        Assert.Equal(42, Functional.Identity(value));
        Assert.Equal("hello", Functional.Identity(text));
    }

    [Fact]
    public void Constant_ShouldReturnSameValue()
    {
        // Arrange
        var alwaysFive = Functional.Constant(5);

        // Act & Assert
        Assert.Equal(5, alwaysFive(10));
        Assert.Equal(5, alwaysFive(100));
        Assert.Equal(5, alwaysFive(0));
    }

    [Fact]
    public void Constant_ShouldWorkWithDifferentTypes()
    {
        // Arrange
        var alwaysHello = Functional.Constant("hello");

        // Act & Assert
        Assert.Equal("hello", alwaysHello(1));
        Assert.Equal("hello", alwaysHello("world"));
    }
}
```

### 6. Tap Tests

```csharp
public class CompositionTests
{
    [Fact]
    public void Tap_ShouldExecuteActionAndReturnValue()
    {
        // Arrange
        int value = 5;
        bool actionExecuted = false;

        // Act
        var result = value.Tap(x => actionExecuted = true);

        // Assert
        Assert.True(actionExecuted);
        Assert.Equal(5, result);  // Original value preserved
    }

    [Fact]
    public void Tap_ShouldPassCorrectValue()
    {
        // Arrange
        int value = 10;
        int passedValue = 0;

        // Act
        var result = value.Tap(x => passedValue = x);

        // Assert
        Assert.Equal(10, passedValue);
        Assert.Equal(10, result);
    }

    [Fact]
    public void Tap_Chaining_ShouldExecuteAllActions()
    {
        // Arrange
        int value = 5;
        var executedActions = new List<int>();

        // Act
        var result = value
            .Tap(x => executedActions.Add(x * 1))
            .Tap(x => executedActions.Add(x * 2))
            .Tap(x => executedActions.Add(x * 3));

        // Assert
        Assert.Equal(3, executedActions.Count);
        Assert.Equal(5, executedActions[0]);
        Assert.Equal(10, executedActions[1]);
        Assert.Equal(15, executedActions[2]);
        Assert.Equal(5, result);  // Original value preserved
    }

    [Fact]
    public void TapIf_True_ShouldExecuteAction()
    {
        // Arrange
        int value = 5;
        bool executed = false;

        // Act
        var result = value.TapIf(true, x => executed = true);

        // Assert
        Assert.True(executed);
        Assert.Equal(5, result);
    }

    [Fact]
    public void TapIf_False_ShouldNotExecuteAction()
    {
        // Arrange
        int value = 5;
        bool executed = false;

        // Act
        var result = value.TapIf(false, x => executed = true);

        // Assert
        Assert.False(executed);
        Assert.Equal(5, result);
    }

    [Fact]
    public void TapForLogging_ShouldLogAndContinue()
    {
        // Arrange
        var value = Tensor.FromArray(new[] { 1f, 2f, 3f });
        var loggedShapes = new List<string>();

        // Act
        var result = value.Tap(x => loggedShapes.Add(x.Shape.ToString()));

        // Assert
        Assert.Single(loggedShapes);
        Assert.Contains("3", loggedShapes[0]);
        Assert.Equal(new[] { 3 }, result.Shape);
    }
}
```

### 7. Real-World Scenario Tests

```csharp
public class CompositionTests
{
    [Fact]
    public void Compose_MLPipeline_ShouldWork()
    {
        // Arrange
        Func<Tensor, Tensor> normalize = t => (t - t.Mean()) / t.Std();
        Func<Tensor, Tensor> applyModel = t => t.MatMul(Tensor.Ones(new[] { 3, 1 }));
        Func<Tensor, Tensor> softmax = t => t.Exp() / t.Exp().Sum();

        // Act
        var pipeline = Functional.Compose(softmax, applyModel, normalize);

        var input = Tensor.FromArray(new[] { 1f, 2f, 3f, 4, 5, 6 }).Reshape(2, 3);
        var result = pipeline(input);

        // Assert
        Assert.Equal(new[] { 2, 1 }, result.Shape);
    }

    [Fact]
    public void Pipe_DataProcessing_ShouldWork()
    {
        // Arrange
        var rawData = new[] { "1", "2", "3", "4", "5" };

        // Act
        var result = rawData
            .Pipe(data => data.Select(int.Parse).ToArray())
            .Pipe(data => data.Where(x => x > 2).ToArray())
            .Pipe(data => data.Sum());

        // Assert
        Assert.Equal(12, result);  // 3 + 4 + 5
    }

    [Fact]
    public void Partial_WithVectorization_ShouldWork()
    {
        // Arrange
        Func<Tensor, Tensor, Tensor> dotProduct = (a, b) => (a * b).Sum();
        var fixedVector = Tensor.FromArray(new[] { 1f, 2f, 3f });

        // Act
        var dotWithFixed = Functional.Partial(dotProduct, fixedVector);
        var batchedDot = Functional.Vectorize(dotWithFixed);

        var batch = Tensor.FromArray(new float[9]).Reshape(3, 3);
        var result = batchedDot(batch);

        // Assert
        Assert.Equal(new[] { 3 }, result.Shape);
    }

    [Fact]
    public void Curry_ComposeCombine_ShouldWork()
    {
        // Arrange
        Func<int, int, int> multiply = (a, b) => a * b;
        Func<int, int, int> add = (a, b) => a + b;

        var curriedMult = Functional.Curry(multiply);
        var curriedAdd = Functional.Curry(add);

        // Act: Create function that multiplies by 2 then adds 3
        var multiplyBy2 = curriedMult(2);
        var add3 = curriedAdd(3);
        var combined = Functional.Compose(add3, multiplyBy2);

        // Assert
        Assert.Equal(13, combined(5));  // (5 * 2) + 3
    }
}
```

## Files to Create
1. `tests/MLFramework.Tests/Functional/CompositionTests.cs`

## Dependencies
- spec_functional_composition.md (implementation must be complete)
- MLFramework.Tensor (for tensor-related tests)

## Success Criteria
- All tests pass
- Compose executes right-to-left correctly
- Pipe executes left-to-right correctly
- Partial application binds arguments correctly
- Currying transforms functions correctly
- Utility functions work as expected
- Real-world scenarios demonstrate usefulness

## Notes for Coder
- Use xUnit for testing
- Test with simple types (int, string) for basic functionality
- Test with Tensor types for ML-specific scenarios
- Ensure generic type handling is correct
- Test edge cases: empty arrays, single functions, null inputs
- Verify that functions are not mutated during composition
- Test that Tap's side effects are executed in order

## Additional Considerations
- Add performance benchmarks for composed vs non-composed functions
- Test thread safety if functions are called concurrently
- Consider adding more advanced composition patterns (Kleisli composition)
- Add tests for composing transformations (vmap, pmap, jit)
- Test memory usage with deep composition chains
