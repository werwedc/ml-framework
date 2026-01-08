using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.HAL.CUDA;

namespace MLFramework.Tests.HAL.CUDA.Graphs;

/// <summary>
/// Unit tests for CUDA Graph core interfaces
/// </summary>
[TestClass]
public class CUDAGraphCoreInterfacesTests
{
    [TestMethod]
    public void CUDAGraphState_Enum_Values_AreCorrect()
    {
        Assert.AreEqual(6, Enum.GetValues(typeof(CUDAGraphState)).Length);
    }

    [TestMethod]
    public void CUDAGraphState_Enum_HasCreatedState()
    {
        Assert.IsTrue(Enum.IsDefined(typeof(CUDAGraphState), CUDAGraphState.Created));
    }

    [TestMethod]
    public void CUDAGraphState_Enum_HasCapturingState()
    {
        Assert.IsTrue(Enum.IsDefined(typeof(CUDAGraphState), CUDAGraphState.Capturing));
    }

    [TestMethod]
    public void CUDAGraphState_Enum_HasReadyState()
    {
        Assert.IsTrue(Enum.IsDefined(typeof(CUDAGraphState), CUDAGraphState.Ready));
    }

    [TestMethod]
    public void CUDAGraphState_Enum_HasExecutingState()
    {
        Assert.IsTrue(Enum.IsDefined(typeof(CUDAGraphState), CUDAGraphState.Executing));
    }

    [TestMethod]
    public void CUDAGraphState_Enum_HasInvalidatedState()
    {
        Assert.IsTrue(Enum.IsDefined(typeof(CUDAGraphState), CUDAGraphState.Invalidated));
    }

    [TestMethod]
    public void CUDAGraphState_Enum_HasDisposedState()
    {
        Assert.IsTrue(Enum.IsDefined(typeof(CUDAGraphState), CUDAGraphState.Disposed));
    }

    [TestMethod]
    public void CUDAGraphValidationResult_Valid_ReturnsTrue()
    {
        var result = new CUDAGraphValidationResult
        {
            IsValid = true,
            Errors = Array.Empty<string>(),
            Warnings = Array.Empty<string>(),
            OperationCount = 100
        };

        Assert.IsTrue(result.IsValid);
        Assert.AreEqual(0, result.Errors.Count);
        Assert.AreEqual(0, result.Warnings.Count);
        Assert.AreEqual(100, result.OperationCount);
    }

    [TestMethod]
    public void CUDAGraphValidationResult_Invalid_ReturnsFalse()
    {
        var result = new CUDAGraphValidationResult
        {
            IsValid = false,
            Errors = new[] { "Error 1", "Error 2" },
            Warnings = new[] { "Warning 1" },
            OperationCount = 0
        };

        Assert.IsFalse(result.IsValid);
        Assert.AreEqual(2, result.Errors.Count);
        Assert.AreEqual(1, result.Warnings.Count);
    }

    [TestMethod]
    public void CUDAGraphValidationResult_Success_ReturnsValidResult()
    {
        var result = CUDAGraphValidationResult.Success(100);

        Assert.IsTrue(result.IsValid);
        Assert.AreEqual(0, result.Errors.Count);
        Assert.AreEqual(0, result.Warnings.Count);
        Assert.AreEqual(100, result.OperationCount);
    }

    [TestMethod]
    public void CUDAGraphValidationResult_Failure_ReturnsInvalidResult()
    {
        var errors = new[] { "Error 1", "Error 2" };
        var result = CUDAGraphValidationResult.Failure(errors, 0);

        Assert.IsFalse(result.IsValid);
        Assert.AreEqual(2, result.Errors.Count);
        Assert.AreEqual(0, result.Warnings.Count);
        Assert.AreEqual(0, result.OperationCount);
    }

    [TestMethod]
    public void CUDAGraphValidationResult_WithWarnings_AddsWarnings()
    {
        var result = CUDAGraphValidationResult.Success(50);
        var warnings = new[] { "Warning 1", "Warning 2" };
        var resultWithWarnings = result.WithWarnings(warnings);

        Assert.IsTrue(resultWithWarnings.IsValid);
        Assert.AreEqual(0, resultWithWarnings.Errors.Count);
        Assert.AreEqual(2, resultWithWarnings.Warnings.Count);
        Assert.AreEqual(50, resultWithWarnings.OperationCount);
    }

    [TestMethod]
    public void CUDAGraphValidationResult_WithWarnings_PreservesValidity()
    {
        var errors = new[] { "Error 1" };
        var result = CUDAGraphValidationResult.Failure(errors, 0);
        var warnings = new[] { "Warning 1" };
        var resultWithWarnings = result.WithWarnings(warnings);

        Assert.IsFalse(resultWithWarnings.IsValid);
        Assert.AreEqual(1, resultWithWarnings.Errors.Count);
        Assert.AreEqual(1, resultWithWarnings.Warnings.Count);
    }

    [TestMethod]
    public void CUDAGraphValidationResult_WithWarnings_EmptyWarningsList()
    {
        var result = CUDAGraphValidationResult.Success(100);
        var resultWithWarnings = result.WithWarnings(Array.Empty<string>());

        Assert.IsTrue(resultWithWarnings.IsValid);
        Assert.AreEqual(0, resultWithWarnings.Warnings.Count);
    }
}
