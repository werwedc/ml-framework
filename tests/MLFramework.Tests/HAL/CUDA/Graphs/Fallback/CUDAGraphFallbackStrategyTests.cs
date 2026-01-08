using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.HAL.CUDA;
using MLFramework.HAL.CUDA.Graphs;
using Moq;
using System;

namespace MLFramework.Tests.HAL.CUDA.Graphs;

/// <summary>
/// Unit tests for CUDA Graph fallback strategy enum
/// </summary>
[TestClass]
public class CUDAGraphFallbackStrategyTests
{
    [TestMethod]
    public void CUDAGraphFallbackStrategy_Enum_HasAllValues()
    {
        Assert.AreEqual(5, Enum.GetValues(typeof(CUDAGraphFallbackStrategy)).Length);
    }

    [TestMethod]
    public void CUDAGraphFallbackStrategy_Enum_HasNeverCapture()
    {
        Assert.IsTrue(Enum.IsDefined(typeof(CUDAGraphFallbackStrategy), CUDAGraphFallbackStrategy.NeverCapture));
    }

    [TestMethod]
    public void CUDAGraphFallbackStrategy_Enum_HasCaptureOrFallback()
    {
        Assert.IsTrue(Enum.IsDefined(typeof(CUDAGraphFallbackStrategy), CUDAGraphFallbackStrategy.CaptureOrFallback));
    }

    [TestMethod]
    public void CUDAGraphFallbackStrategy_Enum_HasCaptureOnly()
    {
        Assert.IsTrue(Enum.IsDefined(typeof(CUDAGraphFallbackStrategy), CUDAGraphFallbackStrategy.CaptureOnly));
    }

    [TestMethod]
    public void CUDAGraphFallbackStrategy_Enum_HasTryOnceThenFallback()
    {
        Assert.IsTrue(Enum.IsDefined(typeof(CUDAGraphFallbackStrategy), CUDAGraphFallbackStrategy.TryOnceThenFallback));
    }

    [TestMethod]
    public void CUDAGraphFallbackStrategy_Enum_HasRetryThenFallback()
    {
        Assert.IsTrue(Enum.IsDefined(typeof(CUDAGraphFallbackStrategy), CUDAGraphFallbackStrategy.RetryThenFallback));
    }
}
