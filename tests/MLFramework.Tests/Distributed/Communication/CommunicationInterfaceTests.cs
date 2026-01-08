namespace MLFramework.Tests.Distributed.Communication;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Distributed.Communication;
using System;

[TestClass]
public class CommunicationInterfaceTests
{
    [TestMethod]
    public void TestReduceOperation_Values()
    {
        // Test that ReduceOperation enum has expected values
        Assert.AreEqual(0, (int)ReduceOperation.Sum);
        Assert.AreEqual(1, (int)ReduceOperation.Product);
        Assert.AreEqual(2, (int)ReduceOperation.Max);
        Assert.AreEqual(3, (int)ReduceOperation.Min);
        Assert.AreEqual(4, (int)ReduceOperation.Avg);
    }

    [TestMethod]
    public void TestCommunicationException_Message()
    {
        var ex = new CommunicationException("Test message");
        Assert.AreEqual("Test message", ex.Message);
        Assert.IsNull(ex.Rank);
        Assert.IsNull(ex.BackendName);
    }

    [TestMethod]
    public void TestCommunicationException_WithRankAndBackend()
    {
        var ex = new CommunicationException("Test message", 0, "NCCL");
        Assert.AreEqual("Test message", ex.Message);
        Assert.AreEqual(0, ex.Rank);
        Assert.AreEqual("NCCL", ex.BackendName);
    }

    [TestMethod]
    public void TestCommunicationException_WithInnerException()
    {
        var inner = new Exception("Inner");
        var ex = new CommunicationException("Test message", inner);
        Assert.AreEqual("Test message", ex.Message);
        Assert.AreEqual(inner, ex.InnerException);
    }

    [TestMethod]
    public void TestCommunicationException_AllParameters()
    {
        var inner = new Exception("Inner");
        var ex = new CommunicationException("Test message", inner, 0, "NCCL");
        Assert.AreEqual("Test message", ex.Message);
        Assert.AreEqual(inner, ex.InnerException);
        Assert.AreEqual(0, ex.Rank);
        Assert.AreEqual("NCCL", ex.BackendName);
    }

    [TestMethod]
    public void TestCommunicationTimeoutException()
    {
        var ex = new CommunicationTimeoutException("Timeout", TimeSpan.FromSeconds(5));
        Assert.AreEqual("Timeout", ex.Message);
        Assert.AreEqual(TimeSpan.FromSeconds(5), ex.TimeoutDuration);
    }

    [TestMethod]
    public void TestCommunicationTimeoutException_WithRankAndBackend()
    {
        var ex = new CommunicationTimeoutException("Timeout", TimeSpan.FromSeconds(5), 0, "NCCL");
        Assert.AreEqual("Timeout", ex.Message);
        Assert.AreEqual(TimeSpan.FromSeconds(5), ex.TimeoutDuration);
        Assert.AreEqual(0, ex.Rank);
        Assert.AreEqual("NCCL", ex.BackendName);
    }

    [TestMethod]
    public void TestRankMismatchException()
    {
        var ex = new RankMismatchException("Mismatch", 0, 1);
        Assert.AreEqual("Mismatch", ex.Message);
        Assert.AreEqual(0, ex.ExpectedRank);
        Assert.AreEqual(1, ex.ActualRank);
    }

    [TestMethod]
    public void TestRankMismatchException_WithRankAndBackend()
    {
        var ex = new RankMismatchException("Mismatch", 0, 1, 2, "NCCL");
        Assert.AreEqual("Mismatch", ex.Message);
        Assert.AreEqual(0, ex.ExpectedRank);
        Assert.AreEqual(1, ex.ActualRank);
        Assert.AreEqual(2, ex.Rank);
        Assert.AreEqual("NCCL", ex.BackendName);
    }

    [TestMethod]
    public void TestCommunicationConfig_DefaultValues()
    {
        var config = new CommunicationConfig();
        Assert.AreEqual(300000, config.TimeoutMs);
        Assert.IsFalse(config.EnableLogging);
        Assert.IsTrue(config.UsePinnedMemory);
        Assert.AreEqual(3, config.MaxRetries);
        Assert.AreEqual(100, config.RetryDelayMs);
    }

    [TestMethod]
    public void TestCommunicationConfig_CustomValues()
    {
        var config = new CommunicationConfig
        {
            TimeoutMs = 60000,
            EnableLogging = true,
            UsePinnedMemory = false,
            MaxRetries = 5,
            RetryDelayMs = 200
        };
        Assert.AreEqual(60000, config.TimeoutMs);
        Assert.IsTrue(config.EnableLogging);
        Assert.IsFalse(config.UsePinnedMemory);
        Assert.AreEqual(5, config.MaxRetries);
        Assert.AreEqual(200, config.RetryDelayMs);
    }
}
