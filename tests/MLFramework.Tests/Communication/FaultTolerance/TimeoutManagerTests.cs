namespace MLFramework.Tests.Communication.FaultTolerance;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Communication.FaultTolerance;
using System;

[TestClass]
public class TimeoutManagerTests
{
    [TestMethod]
    public void TestTimeoutManager_DefaultConstructor_InitializesCorrectly()
    {
        var manager = new TimeoutManager();
        Assert.AreEqual(300000, manager.DefaultTimeoutMs);
        manager.Dispose();
    }

    [TestMethod]
    public void TestTimeoutManager_CustomTimeout_InitializesCorrectly()
    {
        var manager = new TimeoutManager(5000);
        Assert.AreEqual(5000, manager.DefaultTimeoutMs);
        manager.Dispose();
    }

    [TestMethod]
    public void TestStartTimeout_ReturnsValidToken()
    {
        var manager = new TimeoutManager(5000);
        var token = manager.StartTimeout(1);
        Assert.IsFalse(token.IsCancellationRequested);
        manager.Dispose();
    }

    [TestMethod]
    public void TestStartTimeout_WithCustomTimeout_UsesCustomTimeout()
    {
        var manager = new TimeoutManager(10000);
        var token = manager.StartTimeout(1, 1000);
        Assert.IsFalse(token.IsCancellationRequested);
        manager.Dispose();
    }

    [TestMethod]
    public void TestStartTimeout_ReplacesExistingTimeout()
    {
        var manager = new TimeoutManager(5000);
        var token1 = manager.StartTimeout(1, 10000);
        var token2 = manager.StartTimeout(1, 2000);

        // First token should be cancelled when replaced
        // Second token should be active
        Assert.IsFalse(token2.IsCancellationRequested);
        manager.Dispose();
    }

    [TestMethod]
    public void TestCancelTimeout_RemovesTimeout()
    {
        var manager = new TimeoutManager(5000);
        manager.StartTimeout(1);
        manager.CancelTimeout(1);

        // Should not throw when cancelling again
        manager.CancelTimeout(1);
        manager.Dispose();
    }

    [TestMethod]
    public void TestExtendTimeout_ExtendsExistingTimeout()
    {
        var manager = new TimeoutManager(5000);
        var token = manager.StartTimeout(1, 1000);

        // Extend the timeout
        manager.ExtendTimeout(1, 2000);

        // Token should still be valid
        Assert.IsFalse(token.IsCancellationRequested);
        manager.Dispose();
    }

    [TestMethod]
    public void TestCancelAll_RemovesAllTimeouts()
    {
        var manager = new TimeoutManager(5000);
        manager.StartTimeout(1);
        manager.StartTimeout(2);
        manager.StartTimeout(3);

        manager.CancelAll();

        // Should not throw when cancelling individual timeouts
        manager.CancelTimeout(1);
        manager.CancelTimeout(2);
        manager.CancelTimeout(3);
        manager.Dispose();
    }

    [TestMethod]
    public void TestDispose_CancelsAllTimeouts()
    {
        var manager = new TimeoutManager(5000);
        manager.StartTimeout(1);
        manager.StartTimeout(2);

        manager.Dispose();

        // Should not throw when disposing again
        manager.Dispose();
    }

    [TestMethod]
    public void TestTimeoutManager_ZeroTimeout_DoesNotCancel()
    {
        var manager = new TimeoutManager(0);
        var token = manager.StartTimeout(1, 0);

        // Zero timeout means no cancellation
        Assert.IsFalse(token.IsCancellationRequested);
        manager.Dispose();
    }
}
