using Xunit;
using MLFramework.Compilation;

namespace MLFramework.Tests.Compilation;

/// <summary>
/// Unit tests for ShapeSignature
/// </summary>
public class ShapeSignatureTests
{
    [Fact]
    public void Create_WithValidParameters_CreatesSignature()
    {
        // Arrange & Act
        var signature = ShapeSignature.Create("Add", new List<int[]> { new[] { 32, 128 }, new[] { 32, 128 } });

        // Assert
        Assert.Equal("Add", signature.OperationName);
        Assert.Equal(2, signature.InputShapes.Length);
        Assert.Equal(128, signature.InputShapes[0][1]);
    }

    [Fact]
    public void Hash_SameShapes_ReturnsSameHash()
    {
        // Arrange
        var sig1 = ShapeSignature.Create("Mul", new List<int[]> { new[] { 64, 256 } });
        var sig2 = ShapeSignature.Create("Mul", new List<int[]> { new[] { 64, 256 } });

        // Assert
        Assert.Equal(sig1.Hash, sig2.Hash);
    }

    [Fact]
    public void Equals_SameShapes_ReturnsTrue()
    {
        // Arrange
        var sig1 = ShapeSignature.Create("Conv2D", new List<int[]> { new[] { 1, 3, 224, 224 } });
        var sig2 = ShapeSignature.Create("Conv2D", new List<int[]> { new[] { 1, 3, 224, 224 } });

        // Assert
        Assert.True(sig1.Equals(sig2));
    }

    [Fact]
    public void Equals_DifferentShapes_ReturnsFalse()
    {
        // Arrange
        var sig1 = ShapeSignature.Create("Conv2D", new List<int[]> { new[] { 1, 3, 224, 224 } });
        var sig2 = ShapeSignature.Create("Conv2D", new List<int[]> { new[] { 1, 3, 112, 112 } });

        // Assert
        Assert.False(sig1.Equals(sig2));
    }

    [Fact]
    public void EqualityOperator_SameSignatures_ReturnsTrue()
    {
        // Arrange
        var sig1 = ShapeSignature.Create("Relu", new List<int[]> { new[] { 32, 512 } });
        var sig2 = ShapeSignature.Create("Relu", new List<int[]> { new[] { 32, 512 } });

        // Assert
        Assert.True(sig1 == sig2);
    }

    [Fact]
    public void ToString_ReturnsReadableString()
    {
        // Arrange
        var signature = ShapeSignature.Create("Add", new List<int[]> { new[] { 32, 128 }, new[] { 32, 128 } });

        // Act
        var result = signature.ToString();

        // Assert
        Assert.Contains("Add", result);
        Assert.Contains("32", result);
        Assert.Contains("128", result);
    }
}
