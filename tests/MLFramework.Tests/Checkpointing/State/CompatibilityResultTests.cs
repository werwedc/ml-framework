namespace MachineLearning.Checkpointing.Tests;

using Xunit;

/// <summary>
/// Tests for CompatibilityResult
/// </summary>
public class CompatibilityResultTests
{
    [Fact]
    public void Constructor_CreatesEmptyResult()
    {
        // Act
        var result = new CompatibilityResult();

        // Assert
        Assert.Empty(result.Errors);
        Assert.Empty(result.Warnings);
        Assert.True(result.IsCompatible);
        Assert.False(result.HasWarnings);
        Assert.Equal(0, result.TotalIssues);
    }

    [Fact]
    public void AddError_WithValidError_AddsToErrors()
    {
        // Arrange
        var result = new CompatibilityResult();

        // Act
        result.AddError("Test error");

        // Assert
        Assert.Single(result.Errors);
        Assert.Contains("Test error", result.Errors);
        Assert.False(result.IsCompatible);
    }

    [Fact]
    public void AddError_WithEmptyError_DoesNotAdd()
    {
        // Arrange
        var result = new CompatibilityResult();

        // Act
        result.AddError("");
        result.AddError("   ");
        result.AddError(null!);

        // Assert
        Assert.Empty(result.Errors);
    }

    [Fact]
    public void AddWarning_WithValidWarning_AddsToWarnings()
    {
        // Arrange
        var result = new CompatibilityResult();

        // Act
        result.AddWarning("Test warning");

        // Assert
        Assert.Single(result.Warnings);
        Assert.Contains("Test warning", result.Warnings);
        Assert.True(result.IsCompatible); // Warnings don't make it incompatible
        Assert.True(result.HasWarnings);
    }

    [Fact]
    public void AddWarning_WithEmptyWarning_DoesNotAdd()
    {
        // Arrange
        var result = new CompatibilityResult();

        // Act
        result.AddWarning("");
        result.AddWarning("   ");
        result.AddWarning(null!);

        // Assert
        Assert.Empty(result.Warnings);
    }

    [Fact]
    public void AddErrors_WithMultipleErrors_AddsAll()
    {
        // Arrange
        var result = new CompatibilityResult();
        var errors = new[] { "Error 1", "Error 2", "Error 3" };

        // Act
        result.AddErrors(errors);

        // Assert
        Assert.Equal(3, result.Errors.Count);
        Assert.True(result.IsCompatible == false);
    }

    [Fact]
    public void AddWarnings_WithMultipleWarnings_AddsAll()
    {
        // Arrange
        var result = new CompatibilityResult();
        var warnings = new[] { "Warning 1", "Warning 2" };

        // Act
        result.AddWarnings(warnings);

        // Assert
        Assert.Equal(2, result.Warnings.Count);
        Assert.True(result.IsCompatible); // Still compatible with only warnings
    }

    [Fact]
    public void Merge_WithOtherResult_MergesBoth()
    {
        // Arrange
        var result1 = new CompatibilityResult();
        result1.AddError("Error 1");
        result1.AddWarning("Warning 1");

        var result2 = new CompatibilityResult();
        result2.AddError("Error 2");
        result2.AddWarning("Warning 2");

        // Act
        result1.Merge(result2);

        // Assert
        Assert.Equal(2, result1.Errors.Count);
        Assert.Equal(2, result1.Warnings.Count);
    }

    [Fact]
    public void Merge_WithNullResult_DoesNotThrow()
    {
        // Arrange
        var result = new CompatibilityResult();
        result.AddError("Error 1");

        // Act
        result.Merge(null!);

        // Assert
        Assert.Single(result.Errors);
    }

    [Fact]
    public void Success_CreatesSuccessfulResult()
    {
        // Act
        var result = CompatibilityResult.Success();

        // Assert
        Assert.True(result.IsCompatible);
        Assert.Empty(result.Errors);
        Assert.Empty(result.Warnings);
    }

    [Fact]
    public void Failure_CreatesFailedResultWithError()
    {
        // Act
        var result = CompatibilityResult.Failure("Test error");

        // Assert
        Assert.False(result.IsCompatible);
        Assert.Single(result.Errors);
        Assert.Contains("Test error", result.Errors);
    }

    [Fact]
    public void Failures_CreatesFailedResultWithMultipleErrors()
    {
        // Arrange
        var errors = new[] { "Error 1", "Error 2", "Error 3" };

        // Act
        var result = CompatibilityResult.Failures(errors);

        // Assert
        Assert.False(result.IsCompatible);
        Assert.Equal(3, result.Errors.Count);
    }

    [Fact]
    public void ToString_WithErrorsAndWarnings_ReturnsFormattedString()
    {
        // Arrange
        var result = new CompatibilityResult();
        result.AddError("Error 1");
        result.AddError("Error 2");
        result.AddWarning("Warning 1");

        // Act
        var str = result.ToString();

        // Assert
        Assert.Contains("INCOMPATIBLE", str);
        Assert.Contains("Errors: 2", str);
        Assert.Contains("Warnings: 1", str);
        Assert.Contains("Error 1", str);
        Assert.Contains("Error 2", str);
        Assert.Contains("Warning 1", str);
    }

    [Fact]
    public void ToString_WithNoIssues_ReturnsSuccessString()
    {
        // Arrange
        var result = CompatibilityResult.Success();

        // Act
        var str = result.ToString();

        // Assert
        Assert.Contains("COMPATIBLE", str);
        Assert.Contains("Errors: 0", str);
        Assert.Contains("Warnings: 0", str);
    }

    [Fact]
    public void TotalIssues_WithMixedIssues_ReturnsCorrectCount()
    {
        // Arrange
        var result = new CompatibilityResult();
        result.AddError("Error 1");
        result.AddError("Error 2");
        result.AddWarning("Warning 1");
        result.AddWarning("Warning 2");
        result.AddWarning("Warning 3");

        // Act
        var total = result.TotalIssues;

        // Assert
        Assert.Equal(5, total);
    }
}
