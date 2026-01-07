# Spec: Basic Tests for Core Interfaces

## Overview
Create unit tests for the core functional transformation interfaces to ensure the foundational infrastructure works correctly.

## Scope
- Test IFunctionalTransformation interface
- Test BaseTransformation class
- Test TransformationContext
- Test TransformationRegistry
- Test TensorFunctionAttribute

## Test Files to Create
1. `tests/MLFramework.Tests/Functional/CoreInterfacesTests.cs`

## Test Requirements

### 1. TransformationRegistry Tests

```csharp
using Xunit;
using MLFramework.Functional;

namespace MLFramework.Tests.Functional
{
    public class TransformationRegistryTests
    {
        [Fact]
        public void Register_ShouldStoreTransformation()
        {
            // Arrange
            Func<Tensor, Tensor> func = t => t;
            var transform = new DummyTransform();

            // Act
            TransformationRegistry.Register(func, transform);

            // Assert
            var transforms = TransformationRegistry.GetTransformations(func);
            Assert.Single(transforms);
            Assert.Same(transform, transforms[0]);
        }

        [Fact]
        public void Register_ShouldAllowMultipleTransformations()
        {
            // Arrange
            Func<Tensor, Tensor> func = t => t;
            var transform1 = new DummyTransform();
            var transform2 = new DummyTransform();

            // Act
            TransformationRegistry.Register(func, transform1);
            TransformationRegistry.Register(func, transform2);

            // Assert
            var transforms = TransformationRegistry.GetTransformations(func);
            Assert.Equal(2, transforms.Count);
        }

        [Fact]
        public void GetTransformations_ShouldReturnEmptyForUnregistered()
        {
            // Arrange
            Func<Tensor, Tensor> func = t => t;

            // Act
            var transforms = TransformationRegistry.GetTransformations(func);

            // Assert
            Assert.Empty(transforms);
        }

        [Fact]
        public void Clear_ShouldRemoveAllTransformations()
        {
            // Arrange
            Func<Tensor, Tensor> func = t => t;
            TransformationRegistry.Register(func, new DummyTransform());

            // Act
            TransformationRegistry.Clear();

            // Assert
            var transforms = TransformationRegistry.GetTransformations(func);
            Assert.Empty(transforms);
        }
    }

    // Dummy transform for testing
    internal class DummyTransform : BaseTransformation
    {
        public DummyTransform() : base("dummy", TransformationType.Composition) { }

        public override Delegate Transform(Delegate original)
        {
            return original;
        }
    }
}
```

### 2. TransformationContext Tests

```csharp
public class TransformationContextTests
{
    [Fact]
    public void Constructor_ShouldInitializeWithDefaults()
    {
        // Act
        var context = new TransformationContext();

        // Assert
        Assert.False(context.DebugMode);
        Assert.NotNull(context.Metadata);
        Assert.Empty(context.Metadata);
        Assert.Null(context.Parent);
    }

    [Fact]
    public void Constructor_WithParent_ShouldSetParent()
    {
        // Arrange
        var parent = new TransformationContext();

        // Act
        var child = new TransformationContext { Parent = parent };

        // Assert
        Assert.Same(parent, child.Parent);
    }

    [Fact]
    public void Metadata_ShouldStoreAndRetrieveValues()
    {
        // Arrange
        var context = new TransformationContext();

        // Act
        context.Metadata["key"] = "value";

        // Assert
        Assert.Equal("value", context.Metadata["key"]);
    }

    [Fact]
    public void DebugMode_ShouldBeSettable()
    {
        // Arrange
        var context = new TransformationContext();

        // Act
        context.DebugMode = true;

        // Assert
        Assert.True(context.DebugMode);
    }
}
```

### 3. BaseTransformation Tests

```csharp
public class BaseTransformationTests
{
    [Fact]
    public void Constructor_ShouldSetProperties()
    {
        // Act
        var context = new TransformationContext();
        var transform = new DummyTransform(context);

        // Assert
        Assert.Equal("dummy", transform.Name);
        Assert.Equal(TransformationType.Composition, transform.Type);
        Assert.Same(context, transform.Context);
    }

    [Fact]
    public void ValidateDelegate_ShouldThrowForNull()
    {
        // Arrange
        var transform = new DummyTransform();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => transform.ValidateDelegate(null));
    }

    [Fact]
    public void ValidateDelegate_ShouldThrowForNonTensorFunction()
    {
        // Arrange
        var transform = new DummyTransform();
        Func<int, int> func = x => x;

        // Act & Assert
        Assert.Throws<ArgumentException>(() => transform.ValidateDelegate(func));
    }

    [Fact]
    public void ValidateDelegate_ShouldAcceptTensorFunction()
    {
        // Arrange
        var transform = new DummyTransform();
        Func<Tensor, Tensor> func = t => t;

        // Act & Assert
        var exception = Record.Exception(() => transform.ValidateDelegate(func));
        Assert.Null(exception);
    }
}
```

### 4. TensorFunctionAttribute Tests

```csharp
public class TensorFunctionAttributeTests
{
    [Fact]
    public void Constructor_ShouldSetDefaults()
    {
        // Act
        var attr = new TensorFunctionAttribute();

        // Assert
        Assert.True(attr.IsPure);
        Assert.Null(attr.InputShapes);
        Assert.Null(attr.OutputShape);
    }

    [Fact]
    public void Properties_ShouldBeSettable()
    {
        // Arrange
        var attr = new TensorFunctionAttribute();

        // Act
        attr.IsPure = false;
        attr.InputShapes = new[] { "[10]", "[20]" };
        attr.OutputShape = "[5]";

        // Assert
        Assert.False(attr.IsPure);
        Assert.Equal(new[] { "[10]", "[20]" }, attr.InputShapes);
        Assert.Equal("[5]", attr.OutputShape);
    }
}
```

### 5. TransformationType Enum Tests

```csharp
public class TransformationTypeTests
{
    [Fact]
    public void ShouldHaveCorrectValues()
    {
        // Assert
        Assert.Equal(0, (int)TransformationType.Vectorization);
        Assert.Equal(1, (int)TransformationType.Parallelization);
        Assert.Equal(2, (int)TransformationType.Compilation);
        Assert.Equal(3, (int)TransformationType.Composition);
    }
}
```

## Files to Create
1. `tests/MLFramework.Tests/Functional/CoreInterfacesTests.cs`

## Dependencies
- spec_functional_core_interfaces.md (implementation must be complete)

## Success Criteria
- All tests pass
- Tests cover all public APIs
- Edge cases are tested (null, empty, invalid inputs)
- Tests are well-documented and maintainable

## Notes for Coder
- Use xUnit for testing
- Create a DummyTransform class for testing BaseTransformation
- Test both success and failure scenarios
- Use Record.Exception for testing exceptions
- Keep tests focused and independent
- Mock Tensor objects if needed (create simple test doubles)

## Additional Considerations
- Test thread safety of TransformationRegistry (ConcurrentDictionary)
- Test that transformations can be composed (basic composition check)
- Verify that metadata is properly propagated through context hierarchy
