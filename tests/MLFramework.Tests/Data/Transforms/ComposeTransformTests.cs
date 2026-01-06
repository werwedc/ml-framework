using MLFramework.Data.Transforms;
using Xunit;

namespace MLFramework.Tests.Data.Transforms;

public class ComposeTransformTests
{
    [Fact]
    public void Constructor_WithEmptyTransforms_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new ComposeTransform(Array.Empty<ITransform>()));
    }

    [Fact]
    public void Constructor_WithNullTransforms_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new ComposeTransform(null!));
    }

    [Fact]
    public void Constructor_WithNullTransformInArray_ThrowsArgumentNullException()
    {
        // Arrange
        var transforms = new ITransform[] { new MockTransform(), null! };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new ComposeTransform(transforms));
    }

    [Fact]
    public void Constructor_WithSingleTransform_CreatesSuccessfully()
    {
        // Arrange
        var transform = new MockTransform();

        // Act
        var compose = new ComposeTransform(transform);

        // Assert
        Assert.NotNull(compose);
    }

    [Fact]
    public void Constructor_WithMultipleTransforms_CreatesSuccessfully()
    {
        // Arrange
        var transforms = new ITransform[]
        {
            new MockTransform(),
            new MockTransform(),
            new MockTransform()
        };

        // Act
        var compose = new ComposeTransform(transforms);

        // Assert
        Assert.NotNull(compose);
    }

    [Fact]
    public void Apply_WithSingleTransform_AppliesTransformOnce()
    {
        // Arrange
        var mockTransform = new MockTransform();
        var compose = new ComposeTransform(mockTransform);
        var input = "input";

        // Act
        compose.Apply(input);

        // Assert
        Assert.Single(mockTransform.ApplyCalls);
        Assert.Equal(input, mockTransform.ApplyCalls[0]);
    }

    [Fact]
    public void Apply_WithMultipleTransforms_AppliesTransformsInOrder()
    {
        // Arrange
        var transform1 = new MockTransform();
        var transform2 = new MockTransform();
        var transform3 = new MockTransform();
        var compose = new ComposeTransform(transform1, transform2, transform3);
        var input = "input";

        // Act
        compose.Apply(input);

        // Assert
        Assert.Single(transform1.ApplyCalls);
        Assert.Single(transform2.ApplyCalls);
        Assert.Single(transform3.ApplyCalls);

        // Verify order: input -> T1 -> T2 -> T3
        Assert.Equal(input, transform1.ApplyCalls[0]);
        Assert.Equal("transformed", transform2.ApplyCalls[0]);
        Assert.Equal("transformed", transform3.ApplyCalls[0]);
    }

    [Fact]
    public void Apply_WithTransformChain_PassesOutputAsInputToNext()
    {
        // Arrange
        var addOneTransform = new AddOneTransform();
        var multiplyByTwoTransform = new MultiplyByTwoTransform();
        var compose = new ComposeTransform(addOneTransform, multiplyByTwoTransform);
        var input = 5;

        // Act
        var result = compose.Apply(input);

        // Assert
        // 5 + 1 = 6, then 6 * 2 = 12
        Assert.Equal(12, result);
    }

    [Fact]
    public void Apply_WithComplexChain_HandlesSequentialTransforms()
    {
        // Arrange
        var transforms = new ITransform[]
        {
            new AddOneTransform(),
            new MultiplyByTwoTransform(),
            new AddOneTransform(),
            new MultiplyByTwoTransform()
        };
        var compose = new ComposeTransform(transforms);
        var input = 2;

        // Act
        var result = compose.Apply(input);

        // Assert
        // 2 + 1 = 3, 3 * 2 = 6, 6 + 1 = 7, 7 * 2 = 14
        Assert.Equal(14, result);
    }

    // Typed version tests

    [Fact]
    public void TypedConstructor_WithEmptyTransforms_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new ComposeTransform<int, int>(Array.Empty<ITransform>()));
    }

    [Fact]
    public void TypedConstructor_WithNullTransforms_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new ComposeTransform<int, int>(null!));
    }

    [Fact]
    public void TypedConstructor_WithNullTransformInArray_ThrowsArgumentNullException()
    {
        // Arrange
        var transforms = new ITransform[] { new MockTransform(), null! };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new ComposeTransform<int, int>(transforms));
    }

    [Fact]
    public void TypedConstructor_WithSingleTransform_CreatesSuccessfully()
    {
        // Arrange
        var transform = new MockTransform();

        // Act
        var compose = new ComposeTransform<int, int>(transform);

        // Assert
        Assert.NotNull(compose);
    }

    [Fact]
    public void TypedConstructor_WithMultipleTransforms_CreatesSuccessfully()
    {
        // Arrange
        var transforms = new ITransform[]
        {
            new MockTransform(),
            new MockTransform(),
            new MockTransform()
        };

        // Act
        var compose = new ComposeTransform<int, int>(transforms);

        // Assert
        Assert.NotNull(compose);
    }

    [Fact]
    public void TypedApply_WithSingleTransform_AppliesTransformOnce()
    {
        // Arrange
        var mockTransform = new MockTransform();
        var compose = new ComposeTransform<int, int>(mockTransform);
        var input = 5;

        // Act
        compose.Apply(input);

        // Assert
        Assert.Single(mockTransform.ApplyCalls);
        Assert.Equal(input, mockTransform.ApplyCalls[0]);
    }

    [Fact]
    public void TypedApply_WithMultipleTransforms_AppliesTransformsInOrder()
    {
        // Arrange
        var transform1 = new MockTransform();
        var transform2 = new MockTransform();
        var transform3 = new MockTransform();
        var compose = new ComposeTransform<int, int>(transform1, transform2, transform3);
        var input = 5;

        // Act
        compose.Apply(input);

        // Assert
        Assert.Single(transform1.ApplyCalls);
        Assert.Single(transform2.ApplyCalls);
        Assert.Single(transform3.ApplyCalls);
    }

    [Fact]
    public void TypedApply_WithTransformChain_PassesOutputAsInputToNext()
    {
        // Arrange
        var addOneTransform = new AddOneTransform();
        var multiplyByTwoTransform = new MultiplyByTwoTransform();
        var compose = new ComposeTransform<int, int>(addOneTransform, multiplyByTwoTransform);
        var input = 5;

        // Act
        var result = compose.Apply(input);

        // Assert
        // 5 + 1 = 6, then 6 * 2 = 12
        Assert.Equal(12, result);
    }

    [Fact]
    public void TypedApply_WithComplexChain_HandlesSequentialTransforms()
    {
        // Arrange
        var transforms = new ITransform[]
        {
            new AddOneTransform(),
            new MultiplyByTwoTransform(),
            new AddOneTransform(),
            new MultiplyByTwoTransform()
        };
        var compose = new ComposeTransform<int, int>(transforms);
        var input = 2;

        // Act
        var result = compose.Apply(input);

        // Assert
        // 2 + 1 = 3, 3 * 2 = 6, 6 + 1 = 7, 7 * 2 = 14
        Assert.Equal(14, result);
    }

    [Fact]
    public void TypedApply_WithTypeMismatch_ThrowsInvalidCastException()
    {
        // Arrange
        var intTransform = new AddOneTransform();
        var compose = new ComposeTransform<int, string>(intTransform);

        // Act & Assert
        Assert.Throws<InvalidCastException>(() => compose.Apply(5));
    }

    // Helper classes for testing

    private class MockTransform : ITransform
    {
        public List<object> ApplyCalls { get; } = new List<object>();

        public object Apply(object input)
        {
            ApplyCalls.Add(input);
            return "transformed";
        }
    }

    private class AddOneTransform : ITransform
    {
        public object Apply(object input)
        {
            return (int)input + 1;
        }
    }

    private class MultiplyByTwoTransform : ITransform
    {
        public object Apply(object input)
        {
            return (int)input * 2;
        }
    }
}
