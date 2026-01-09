using MLFramework.Diagnostics;
using RitterFramework.Core;
using Xunit;

namespace MLFramework.Tests.Diagnostics;

/// <summary>
/// Unit tests for diagnostic formatters.
/// </summary>
public class DiagnosticFormatterTests
{
    private readonly DiagnosticFormatterRegistry _registry;

    public DiagnosticFormatterTests()
    {
        _registry = DiagnosticFormatterRegistry.Instance;

        // Register all formatters
        _registry.Register(new MatrixMultiplyDiagnosticFormatter());
        _registry.Register(new Conv2DDiagnosticFormatter());
        _registry.Register(new ConcatDiagnosticFormatter());
        _registry.Register(new BroadcastDiagnosticFormatter());
    }

    [Fact]
    public void MatrixMultiplyFormatter_FormatsErrorCorrectly()
    {
        // Arrange
        var formatter = new MatrixMultiplyDiagnosticFormatter();
        var result = ValidationResult.Invalid("Shape mismatch");
        var inputShapes = new long[][]
        {
            new long[] { 32, 256 },
            new long[] { 128, 10 }
        };

        // Act
        var error = formatter.FormatError(result, inputShapes);

        // Assert
        Assert.Contains("Matrix multiplication", error);
        Assert.Contains("[32, 256]", error);
        Assert.Contains("[128, 10]", error);
        Assert.Contains("Inner dimensions 256 and 128 must match", error);
    }

    [Fact]
    public void MatrixMultiplyFormatter_GeneratesRelevantSuggestions()
    {
        // Arrange
        var formatter = new MatrixMultiplyDiagnosticFormatter();
        var result = ValidationResult.Invalid("Shape mismatch with 256 and 128");

        // Act
        var suggestions = formatter.GenerateSuggestions(result);

        // Assert
        Assert.NotNull(suggestions);
        Assert.NotEmpty(suggestions);
        Assert.Contains("Check layer configurations", string.Join(" ", suggestions));
    }

    [Fact]
    public void Conv2DFormatter_FormatsErrorCorrectly()
    {
        // Arrange
        var formatter = new Conv2DDiagnosticFormatter();
        var result = ValidationResult.Invalid("Channel mismatch");
        var inputShapes = new long[][]
        {
            new long[] { 32, 3, 224, 224 },
            new long[] { 64, 3, 3, 3 }
        };

        // Act
        var error = formatter.FormatError(result, inputShapes);

        // Assert
        Assert.Contains("Conv2D", error);
        Assert.Contains("[32, 3, 224, 224]", error);
        Assert.Contains("[64, 3, 3, 3]", error);
        Assert.Contains("Input channels (3) match kernel channels (3)", error);
    }

    [Fact]
    public void Conv2DFormatter_GeneratesRelevantSuggestions()
    {
        // Arrange
        var formatter = new Conv2DDiagnosticFormatter();
        var result = ValidationResult.Invalid("Channel configuration error");

        // Act
        var suggestions = formatter.GenerateSuggestions(result);

        // Assert
        Assert.NotNull(suggestions);
        Assert.NotEmpty(suggestions);
        Assert.Contains("Check channel configurations", string.Join(" ", suggestions));
    }

    [Fact]
    public void ConcatFormatter_FormatsErrorCorrectly()
    {
        // Arrange
        var formatter = new ConcatDiagnosticFormatter();
        var result = ValidationResult.Invalid("Valid concatenation");
        var inputShapes = new long[][]
        {
            new long[] { 32, 128 },
            new long[] { 32, 64 }
        };

        // Act
        var error = formatter.FormatError(result, inputShapes);

        // Assert
        Assert.Contains("Concat", error);
        Assert.Contains("[32, 128]", error);
        Assert.Contains("[32, 64]", error);
        Assert.Contains("Valid: All dimensions match except on axis 1", error);
        Assert.Contains("[32, 192]", error); // Output shape
    }

    [Fact]
    public void ConcatFormatter_GeneratesRelevantSuggestions()
    {
        // Arrange
        var formatter = new ConcatDiagnosticFormatter();
        var result = ValidationResult.Invalid("Axis mismatch");

        // Act
        var suggestions = formatter.GenerateSuggestions(result);

        // Assert
        Assert.NotNull(suggestions);
        Assert.NotEmpty(suggestions);
        Assert.Contains("Check tensor shapes", string.Join(" ", suggestions));
    }

    [Fact]
    public void BroadcastFormatter_FormatsErrorCorrectly()
    {
        // Arrange
        var formatter = new BroadcastDiagnosticFormatter();
        var result = ValidationResult.Invalid("Broadcasting failed");
        var inputShapes = new long[][]
        {
            new long[] { 32, 1 },
            new long[] { 32, 10 }
        };

        // Act
        var error = formatter.FormatError(result, inputShapes);

        // Assert
        Assert.Contains("Broadcast", error);
        Assert.Contains("[32, 1]", error);
        Assert.Contains("[32, 10]", error);
    }

    [Fact]
    public void BroadcastFormatter_GeneratesRelevantSuggestions()
    {
        // Arrange
        var formatter = new BroadcastDiagnosticFormatter();
        var result = ValidationResult.Invalid("Dimension incompatible");

        // Act
        var suggestions = formatter.GenerateSuggestions(result);

        // Assert
        Assert.NotNull(suggestions);
        Assert.NotEmpty(suggestions);
        Assert.Contains("Check broadcasting rules", string.Join(" ", suggestions));
    }

    [Fact]
    public void Registry_FormatError_UsesCorrectFormatter()
    {
        // Arrange
        var result = ValidationResult.Invalid("Shape mismatch");
        var matrixMultiplyShapes = new long[][]
        {
            new long[] { 32, 256 },
            new long[] { 128, 10 }
        };

        // Act
        var error = _registry.FormatError(OperationType.MatrixMultiply, result, matrixMultiplyShapes);

        // Assert
        Assert.Contains("Matrix multiplication", error);
        Assert.Contains("Inner dimensions 256 and 128 must match", error);
    }

    [Fact]
    public void Registry_FormatError_UseGenericMessage_WhenNoFormatterRegistered()
    {
        // Arrange
        var result = ValidationResult.Invalid("Some error");
        var shapes = new long[][] { new long[] { 1, 2, 3 } };

        // Act
        var error = _registry.FormatError(OperationType.Transpose, result, shapes);

        // Assert
        Assert.Contains("Transpose", error);
        Assert.Contains("Validation failed", error);
    }

    [Fact]
    public void Registry_GetSuggestions_UsesCorrectFormatter()
    {
        // Arrange
        var result = ValidationResult.Invalid("Shape mismatch");

        // Act
        var suggestions = _registry.GetSuggestions(OperationType.MatrixMultiply, result);

        // Assert
        Assert.NotNull(suggestions);
        Assert.NotEmpty(suggestions);
        Assert.Contains("Check layer configurations", string.Join(" ", suggestions));
    }

    [Fact]
    public void Registry_GetSuggestions_ReturnsEmptyList_WhenNoFormatterRegistered()
    {
        // Arrange
        var result = ValidationResult.Invalid("Some error");

        // Act
        var suggestions = _registry.GetSuggestions(OperationType.Transpose, result);

        // Assert
        Assert.NotNull(suggestions);
        Assert.Empty(suggestions);
    }

    [Fact]
    public void ConcatFormatter_DetectsIncompatibleShapes()
    {
        // Arrange
        var formatter = new ConcatDiagnosticFormatter();
        var result = ValidationResult.Invalid("Shapes are incompatible");
        var inputShapes = new long[][]
        {
            new long[] { 32, 128 },
            new long[] { 64, 128 } // Different batch size
        };

        // Act
        var error = formatter.FormatError(result, inputShapes);

        // Assert
        Assert.Contains("Concat", error);
        Assert.Contains("[32, 128]", error);
        Assert.Contains("[64, 128]", error);
        // Should not say "Valid" since shapes are incompatible
        Assert.DoesNotContain("Valid: All dimensions match", error);
    }

    [Fact]
    public void BroadcastFormatter_DetectsIncompatibleDimensions()
    {
        // Arrange
        var formatter = new BroadcastDiagnosticFormatter();
        var result = ValidationResult.Invalid("Cannot broadcast");
        var inputShapes = new long[][]
        {
            new long[] { 32, 10 },
            new long[] { 20, 10 } // Incompatible batch sizes
        };

        // Act
        var error = formatter.FormatError(result, inputShapes);

        // Assert
        Assert.Contains("Broadcast", error);
        Assert.Contains("[32, 10]", error);
        Assert.Contains("[20, 10]", error);
        // Should indicate dimension problem
        Assert.Contains("Problem", error);
    }

    [Fact]
    public void BroadcastFormatter_ValidatesCompatibleShapes()
    {
        // Arrange
        var formatter = new BroadcastDiagnosticFormatter();
        var result = ValidationResult.Invalid("Testing compatibility");
        var inputShapes = new long[][]
        {
            new long[] { 32, 1 },
            new long[] { 32, 10 } // Can broadcast dimension 1
        };

        // Act
        var error = formatter.FormatError(result, inputShapes);

        // Assert
        Assert.Contains("Valid: Shapes are compatible for broadcasting", error);
    }

    [Fact]
    public void Formatter_ThrowsArgumentNullException_WhenFormatterIsNull()
    {
        // Arrange
        IDiagnosticFormatter? nullFormatter = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => _registry.Register(nullFormatter!));
    }

    [Fact]
    public void MatrixMultiplyFormatter_HandlesInsufficientInputShapes()
    {
        // Arrange
        var formatter = new MatrixMultiplyDiagnosticFormatter();
        var result = ValidationResult.Invalid("Test error");
        var inputShapes = new long[][] { new long[] { 32, 256 } }; // Only one shape

        // Act
        var error = formatter.FormatError(result, inputShapes);

        // Assert
        Assert.Contains("Insufficient input shapes provided", error);
    }

    [Fact]
    public void Conv2DFormatter_HandlesInsufficientInputShapes()
    {
        // Arrange
        var formatter = new Conv2DDiagnosticFormatter();
        var result = ValidationResult.Invalid("Test error");
        var inputShapes = new long[][] { new long[] { 32, 3, 224, 224 } }; // Only one shape

        // Act
        var error = formatter.FormatError(result, inputShapes);

        // Assert
        Assert.Contains("Insufficient input shapes provided", error);
    }

    [Fact]
    public void ConcatFormatter_HandlesInsufficientInputShapes()
    {
        // Arrange
        var formatter = new ConcatDiagnosticFormatter();
        var result = ValidationResult.Invalid("Test error");
        var inputShapes = new long[][] { new long[] { 32, 128 } }; // Only one shape

        // Act
        var error = formatter.FormatError(result, inputShapes);

        // Assert
        Assert.Contains("Insufficient input shapes provided", error);
    }

    [Fact]
    public void BroadcastFormatter_HandlesInsufficientInputShapes()
    {
        // Arrange
        var formatter = new BroadcastDiagnosticFormatter();
        var result = ValidationResult.Invalid("Test error");
        var inputShapes = new long[][] { new long[] { 32, 1 } }; // Only one shape

        // Act
        var error = formatter.FormatError(result, inputShapes);

        // Assert
        Assert.Contains("Insufficient input shapes provided", error);
    }
}
