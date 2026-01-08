using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.HAL.CUDA.Graphs;
using System;

namespace MLFramework.Tests.HAL.CUDA.Graphs;

/// <summary>
/// Unit tests for CUDA Graph capture exception
/// </summary>
[TestClass]
public class CUDAGraphCaptureExceptionTests
{
    [TestMethod]
    public void Constructor_WithMessage_SetsMessageCorrectly()
    {
        // Arrange
        var message = "Capture failed";

        // Act
        var exception = new CUDAGraphCaptureException(message);

        // Assert
        Assert.AreEqual(message, exception.Message);
        Assert.IsNull(exception.ValidationResult);
    }

    [TestMethod]
    public void Constructor_WithMessageAndInnerException_SetsPropertiesCorrectly()
    {
        // Arrange
        var message = "Capture failed";
        var innerException = new InvalidOperationException("Inner error");

        // Act
        var exception = new CUDAGraphCaptureException(message, innerException);

        // Assert
        Assert.AreEqual(message, exception.Message);
        Assert.AreEqual(innerException, exception.InnerException);
        Assert.IsNull(exception.ValidationResult);
    }

    [TestMethod]
    public void Constructor_WithMessageAndValidationResult_SetsPropertiesCorrectly()
    {
        // Arrange
        var message = "Validation failed";
        var validationResult = CUDAGraphValidationResult.Failure(new[] { "Error 1", "Error 2" });

        // Act
        var exception = new CUDAGraphCaptureException(message, validationResult);

        // Assert
        Assert.AreEqual(message, exception.Message);
        Assert.AreEqual(validationResult, exception.ValidationResult);
        Assert.IsFalse(validationResult.IsValid);
    }

    [TestMethod]
    public void Constructor_WithAllParameters_SetsPropertiesCorrectly()
    {
        // Arrange
        var message = "Validation failed with inner exception";
        var validationResult = CUDAGraphValidationResult.Failure(new[] { "Error 1" });
        var innerException = new InvalidOperationException("Inner error");

        // Act
        var exception = new CUDAGraphCaptureException(message, validationResult, innerException);

        // Assert
        Assert.AreEqual(message, exception.Message);
        Assert.AreEqual(validationResult, exception.ValidationResult);
        Assert.AreEqual(innerException, exception.InnerException);
        Assert.IsFalse(validationResult.IsValid);
    }

    [TestMethod]
    public void Constructor_WithSuccessValidationResult_AllowsCreation()
    {
        // Arrange
        var message = "Unexpected error";
        var validationResult = CUDAGraphValidationResult.Success(10);

        // Act
        var exception = new CUDAGraphCaptureException(message, validationResult);

        // Assert
        Assert.AreEqual(message, exception.Message);
        Assert.AreEqual(validationResult, exception.ValidationResult);
        Assert.IsTrue(validationResult.IsValid);
    }

    [TestMethod]
    public void CUDAGraphCaptureException_IsAssignableFromException()
    {
        // Arrange
        var exception = new CUDAGraphCaptureException("Test");

        // Assert
        Assert.IsInstanceOfType<Exception>(exception);
    }
}
