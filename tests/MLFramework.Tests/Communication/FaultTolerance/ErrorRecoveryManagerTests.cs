namespace MLFramework.Tests.Communication.FaultTolerance;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Communication.FaultTolerance;
using MLFramework.Distributed.Communication;
using System;
using System.IO;

[TestClass]
public class ErrorRecoveryManagerTests
{
    [TestMethod]
    public void TestErrorRecoveryManager_DefaultConstructor_InitializesCorrectly()
    {
        var manager = new ErrorRecoveryManager();
        Assert.AreEqual(3, manager.MaxRetries);
        manager.Dispose();
    }

    [TestMethod]
    public void TestErrorRecoveryManager_CustomRetries_InitializesCorrectly()
    {
        var manager = new ErrorRecoveryManager(5, TimeSpan.FromMilliseconds(200));
        Assert.AreEqual(5, manager.MaxRetries);
        manager.Dispose();
    }

    [TestMethod]
    public void TestHandleError_TimeoutException_ReturnsTrueForRetry()
    {
        var manager = new ErrorRecoveryManager(3);
        var error = new CommunicationError
        {
            OperationId = 1,
            OperationType = "TestOp",
            Severity = ErrorSeverity.Recoverable,
            Exception = new CommunicationTimeoutException("Timeout", TimeSpan.FromSeconds(5)),
            Timestamp = DateTime.Now,
            Rank = 0
        };

        var shouldRetry = manager.HandleError(error);
        Assert.IsTrue(shouldRetry);
        manager.Dispose();
    }

    [TestMethod]
    public void TestHandleError_RankMismatchException_ReturnsFalseForAbort()
    {
        var manager = new ErrorRecoveryManager(3);
        var error = new CommunicationError
        {
            OperationId = 1,
            OperationType = "TestOp",
            Severity = ErrorSeverity.Fatal,
            Exception = new RankMismatchException("Mismatch", 0, 1),
            Timestamp = DateTime.Now,
            Rank = 0
        };

        var shouldRetry = manager.HandleError(error);
        Assert.IsFalse(shouldRetry);
        manager.Dispose();
    }

    [TestMethod]
    public void TestHandleError_IOException_ReturnsTrueForRetry()
    {
        var manager = new ErrorRecoveryManager(3);
        var error = new CommunicationError
        {
            OperationId = 1,
            OperationType = "TestOp",
            Severity = ErrorSeverity.Recoverable,
            Exception = new IOException("Network error"),
            Timestamp = DateTime.Now,
            Rank = 0
        };

        var shouldRetry = manager.HandleError(error);
        Assert.IsTrue(shouldRetry);
        manager.Dispose();
    }

    [TestMethod]
    public void TestHandleError_MaxRetriesExceeded_ReturnsFalse()
    {
        var manager = new ErrorRecoveryManager(2);
        var error = new CommunicationError
        {
            OperationId = 1,
            OperationType = "TestOp",
            Severity = ErrorSeverity.Recoverable,
            Exception = new CommunicationTimeoutException("Timeout", TimeSpan.FromSeconds(5)),
            Timestamp = DateTime.Now,
            Rank = 0
        };

        // Retry 1
        var result1 = manager.HandleError(error);
        Assert.IsTrue(result1);

        // Retry 2
        var result2 = manager.HandleError(error);
        Assert.IsTrue(result2);

        // Retry 3 (exceeds max retries)
        var result3 = manager.HandleError(error);
        Assert.IsFalse(result3);
        manager.Dispose();
    }

    [TestMethod]
    public void TestSetRecoveryStrategy_UpdatesStrategy()
    {
        var manager = new ErrorRecoveryManager(3);
        manager.SetRecoveryStrategy(typeof(IOException), RecoveryStrategy.Abort);

        var error = new CommunicationError
        {
            OperationId = 1,
            OperationType = "TestOp",
            Severity = ErrorSeverity.Recoverable,
            Exception = new IOException("Network error"),
            Timestamp = DateTime.Now,
            Rank = 0
        };

        var shouldRetry = manager.HandleError(error);
        Assert.IsFalse(shouldRetry);
        manager.Dispose();
    }

    [TestMethod]
    public void TestGetErrorHistory_ReturnsAllErrors()
    {
        var manager = new ErrorRecoveryManager(3);
        var error1 = new CommunicationError
        {
            OperationId = 1,
            OperationType = "TestOp1",
            Severity = ErrorSeverity.Recoverable,
            Exception = new CommunicationTimeoutException("Timeout", TimeSpan.FromSeconds(5)),
            Timestamp = DateTime.Now,
            Rank = 0
        };

        var error2 = new CommunicationError
        {
            OperationId = 2,
            OperationType = "TestOp2",
            Severity = ErrorSeverity.Fatal,
            Exception = new RankMismatchException("Mismatch", 0, 1),
            Timestamp = DateTime.Now,
            Rank = 0
        };

        manager.HandleError(error1);
        manager.HandleError(error2);

        var history = manager.GetErrorHistory();
        Assert.AreEqual(2, history.Count);
        manager.Dispose();
    }

    [TestMethod]
    public void TestGetStatistics_ReturnsCorrectStatistics()
    {
        var manager = new ErrorRecoveryManager(3);
        var error1 = new CommunicationError
        {
            OperationId = 1,
            OperationType = "TestOp",
            Severity = ErrorSeverity.Recoverable,
            Exception = new CommunicationTimeoutException("Timeout", TimeSpan.FromSeconds(5)),
            Timestamp = DateTime.Now,
            Rank = 0
        };

        var error2 = new CommunicationError
        {
            OperationId = 2,
            OperationType = "TestOp",
            Severity = ErrorSeverity.Recoverable,
            Exception = new CommunicationTimeoutException("Timeout", TimeSpan.FromSeconds(5)),
            Timestamp = DateTime.Now,
            Rank = 0
        };

        manager.HandleError(error1);
        manager.HandleError(error2);

        var stats = manager.GetStatistics();
        Assert.AreEqual(2, stats.TotalErrors);
        Assert.IsTrue(stats.ErrorsByType.ContainsKey("CommunicationTimeoutException"));
        Assert.AreEqual(2, stats.ErrorsByType["CommunicationTimeoutException"]);
        Assert.AreEqual(2, stats.ErrorsBySeverity[ErrorSeverity.Recoverable]);
        Assert.AreEqual(2, stats.ErrorsByOperation["TestOp"]);
        manager.Dispose();
    }

    [TestMethod]
    public void TestClearHistory_RemovesAllErrors()
    {
        var manager = new ErrorRecoveryManager(3);
        var error = new CommunicationError
        {
            OperationId = 1,
            OperationType = "TestOp",
            Severity = ErrorSeverity.Recoverable,
            Exception = new CommunicationTimeoutException("Timeout", TimeSpan.FromSeconds(5)),
            Timestamp = DateTime.Now,
            Rank = 0
        };

        manager.HandleError(error);
        manager.ClearHistory();

        var history = manager.GetErrorHistory();
        Assert.AreEqual(0, history.Count);
        manager.Dispose();
    }

    [TestMethod]
    public void TestDispose_ClearsHistory()
    {
        var manager = new ErrorRecoveryManager(3);
        var error = new CommunicationError
        {
            OperationId = 1,
            OperationType = "TestOp",
            Severity = ErrorSeverity.Recoverable,
            Exception = new CommunicationTimeoutException("Timeout", TimeSpan.FromSeconds(5)),
            Timestamp = DateTime.Now,
            Rank = 0
        };

        manager.HandleError(error);
        manager.Dispose();

        // Should not throw when disposing again
        manager.Dispose();
    }
}
