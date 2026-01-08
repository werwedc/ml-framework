using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace MLFramework.Tests.Serving;

[TestClass]
public class ResponseScattererTests
{
    [TestMethod]
    public void Scatter_WithAllResponses_SetsAllTasks()
    {
        var scatterer = new ResponseScatterer<int>();
        var requests = new[]
        {
            new QueuedRequest<string, int>("test1"),
            new QueuedRequest<string, int>("test2"),
            new QueuedRequest<string, int>("test3")
        };
        var responses = new List<int> { 1, 2, 3 };

        scatterer.Scatter(requests, responses);

        Assert.IsTrue(requests[0].ResponseSource.Task.IsCompleted);
        Assert.IsTrue(requests[1].ResponseSource.Task.IsCompleted);
        Assert.IsTrue(requests[2].ResponseSource.Task.IsCompleted);

        // Verify correct responses
        Assert.AreEqual(1, requests[0].ResponseSource.Task.Result);
        Assert.AreEqual(2, requests[1].ResponseSource.Task.Result);
        Assert.AreEqual(3, requests[2].ResponseSource.Task.Result);
    }

    [TestMethod]
    public void Scatter_WithException_SetsAllTasksToFaulted()
    {
        var scatterer = new ResponseScatterer<int>();
        var requests = new[]
        {
            new QueuedRequest<string, int>("test1"),
            new QueuedRequest<string, int>("test2")
        };
        var exception = new InvalidOperationException("Batch failed");

        scatterer.Scatter(requests, null, exception);

        Assert.IsTrue(requests[0].ResponseSource.Task.IsFaulted);
        Assert.IsTrue(requests[1].ResponseSource.Task.IsFaulted);

        // Verify correct exception
        var ex1 = await Assert.ThrowsExceptionAsync<InvalidOperationException>(() => requests[0].ResponseSource.Task);
        var ex2 = await Assert.ThrowsExceptionAsync<InvalidOperationException>(() => requests[1].ResponseSource.Task);
        Assert.AreEqual("Batch failed", ex1.Message);
        Assert.AreEqual("Batch failed", ex2.Message);
    }

    [TestMethod]
    public void ScatterWithPartialFailures_HandlesMixedResults()
    {
        var scatterer = new ResponseScatterer<int>();
        var requests = new[]
        {
            new QueuedRequest<string, int>("test1"),
            new QueuedRequest<string, int>("test2"),
            new QueuedRequest<string, int>("test3")
        };
        var responses = new List<int> { 1, 0, 3 };
        var exceptions = new List<Exception>
        {
            null,
            new InvalidOperationException("Failed"),
            null
        };

        scatterer.ScatterWithPartialFailures(requests, responses, exceptions);

        Assert.IsTrue(requests[0].ResponseSource.Task.IsCompletedSuccessfully);
        Assert.IsTrue(requests[1].ResponseSource.Task.IsFaulted);
        Assert.IsTrue(requests[2].ResponseSource.Task.IsCompletedSuccessfully);

        // Verify correct responses
        Assert.AreEqual(1, requests[0].ResponseSource.Task.Result);
        Assert.AreEqual(3, requests[2].ResponseSource.Task.Result);

        // Verify exception
        var ex = await Assert.ThrowsExceptionAsync<InvalidOperationException>(() => requests[1].ResponseSource.Task);
        Assert.AreEqual("Failed", ex.Message);
    }

    [TestMethod]
    public void Scatter_WithNullRequests_Throws()
    {
        var scatterer = new ResponseScatterer<int>();
        var responses = new List<int> { 1, 2, 3 };

        Assert.ThrowsException<ArgumentNullException>(() =>
        {
            scatterer.Scatter<string, int>(null, responses);
        });
    }

    [TestMethod]
    public void Scatter_WithNullResponses_Throws()
    {
        var scatterer = new ResponseScatterer<int>();
        var requests = new[]
        {
            new QueuedRequest<string, int>("test1"),
            new QueuedRequest<string, int>("test2")
        };

        Assert.ThrowsException<ArgumentNullException>(() =>
        {
            scatterer.Scatter(requests, null);
        });
    }

    [TestMethod]
    public void Scatter_WithMismatchedCounts_Throws()
    {
        var scatterer = new ResponseScatterer<int>();
        var requests = new[]
        {
            new QueuedRequest<string, int>("test1"),
            new QueuedRequest<string, int>("test2")
        };
        var responses = new List<int> { 1 }; // Only one response for two requests

        var ex = Assert.ThrowsException<ArgumentException>(() =>
        {
            scatterer.Scatter(requests, responses);
        });

        Assert.IsTrue(ex.Message.Contains("does not match request count"));
    }

    [TestMethod]
    public void Scatter_WithEmptyLists_Succeeds()
    {
        var scatterer = new ResponseScatterer<int>();
        var requests = Array.Empty<QueuedRequest<string, int>>();
        var responses = new List<int>();

        scatterer.Scatter(requests, responses);

        // Should complete without throwing
    }

    [TestMethod]
    public void Scatter_PreservesOrder()
    {
        var scatterer = new ResponseScatterer<int>();
        var requests = new[]
        {
            new QueuedRequest<string, int>("a"),
            new QueuedRequest<string, int>("b"),
            new QueuedRequest<string, int>("c"),
            new QueuedRequest<string, int>("d")
        };
        var responses = new List<int> { 10, 20, 30, 40 };

        scatterer.Scatter(requests, responses);

        // Verify order is preserved
        Assert.AreEqual(10, requests[0].ResponseSource.Task.Result);
        Assert.AreEqual(20, requests[1].ResponseSource.Task.Result);
        Assert.AreEqual(30, requests[2].ResponseSource.Task.Result);
        Assert.AreEqual(40, requests[3].ResponseSource.Task.Result);
    }

    [TestMethod]
    public void ScatterWithPartialFailures_WithNullRequests_Throws()
    {
        var scatterer = new ResponseScatterer<int>();
        var responses = new List<int> { 1, 2 };
        var exceptions = new List<Exception> { null, null };

        Assert.ThrowsException<ArgumentNullException>(() =>
        {
            scatterer.ScatterWithPartialFailures<string, int>(null, responses, exceptions);
        });
    }

    [TestMethod]
    public void ScatterWithPartialFailures_WithNullResponses_Throws()
    {
        var scatterer = new ResponseScatterer<int>();
        var requests = new[]
        {
            new QueuedRequest<string, int>("test1"),
            new QueuedRequest<string, int>("test2")
        };
        var exceptions = new List<Exception> { null, null };

        Assert.ThrowsException<ArgumentNullException>(() =>
        {
            scatterer.ScatterWithPartialFailures(requests, null, exceptions);
        });
    }

    [TestMethod]
    public void ScatterWithPartialFailures_WithNullExceptions_Throws()
    {
        var scatterer = new ResponseScatterer<int>();
        var requests = new[]
        {
            new QueuedRequest<string, int>("test1"),
            new QueuedRequest<string, int>("test2")
        };
        var responses = new List<int> { 1, 2 };

        Assert.ThrowsException<ArgumentNullException>(() =>
        {
            scatterer.ScatterWithPartialFailures(requests, responses, null);
        });
    }

    [TestMethod]
    public void ScatterWithPartialFailures_WithMismatchedCounts_Throws()
    {
        var scatterer = new ResponseScatterer<int>();
        var requests = new[]
        {
            new QueuedRequest<string, int>("test1"),
            new QueuedRequest<string, int>("test2")
        };
        var responses = new List<int> { 1 };
        var exceptions = new List<Exception> { null };

        var ex = Assert.ThrowsException<ArgumentException>(() =>
        {
            scatterer.ScatterWithPartialFailures(requests, responses, exceptions);
        });

        Assert.IsTrue(ex.Message.Contains("must match request count"));
    }

    [TestMethod]
    public void ScatterWithPartialFailures_WithNoResponseAndNoException_Throws()
    {
        var scatterer = new ResponseScatterer<int>();
        var requests = new[]
        {
            new QueuedRequest<string, int>("test1")
        };
        var responses = new List<int> { 0 }; // Default value, effectively null
        var exceptions = new List<Exception> { null };

        scatterer.ScatterWithPartialFailures(requests, responses, exceptions);

        // Should fault with InvalidOperationException
        var ex = await Assert.ThrowsExceptionAsync<InvalidOperationException>(() => requests[0].ResponseSource.Task);
        Assert.IsTrue(ex.Message.Contains("No response or exception provided for request"));
    }

    [TestMethod]
    public void Scatter_WithDifferentTypes_WorksCorrectly()
    {
        var scatterer = new ResponseScatterer<string>();
        var requests = new[]
        {
            new QueuedRequest<int, string>(1),
            new QueuedRequest<int, string>(2),
            new QueuedRequest<int, string>(3)
        };
        var responses = new List<string> { "one", "two", "three" };

        scatterer.Scatter(requests, responses);

        Assert.AreEqual("one", requests[0].ResponseSource.Task.Result);
        Assert.AreEqual("two", requests[1].ResponseSource.Task.Result);
        Assert.AreEqual("three", requests[2].ResponseSource.Task.Result);
    }

    [TestMethod]
    public void Scatter_WithComplexResponseTypes_WorksCorrectly()
    {
        var scatterer = new ResponseScatterer<List<int>>();
        var requests = new[]
        {
            new QueuedRequest<string, List<int>>("test1"),
            new QueuedRequest<string, List<int>>("test2")
        };
        var responses = new List<List<int>>
        {
            new List<int> { 1, 2, 3 },
            new List<int> { 4, 5, 6 }
        };

        scatterer.Scatter(requests, responses);

        var r1 = requests[0].ResponseSource.Task.Result;
        var r2 = requests[1].ResponseSource.Task.Result;

        Assert.AreEqual(3, r1.Count);
        Assert.AreEqual(3, r2.Count);
        Assert.AreEqual(1, r1[0]);
        Assert.AreEqual(6, r2[2]);
    }

    [TestMethod]
    public async Task Scatter_WithAlreadyCompletedTask_DoesNotThrow()
    {
        var scatterer = new ResponseScatterer<int>();
        var request = new QueuedRequest<string, int>("test");

        // Manually complete the task first
        request.ResponseSource.TrySetResult(999);

        var requests = new[] { request };
        var responses = new List<int> { 111 };

        // Should handle gracefully (TrySetResult returns false silently)
        scatterer.Scatter(requests, responses);

        // Original value should remain
        Assert.AreEqual(999, request.ResponseSource.Task.Result);
    }

    [TestMethod]
    public void Scatter_BatchLevelException_DistributesToAllRequests()
    {
        var scatterer = new ResponseScatterer<int>();
        var requests = new[]
        {
            new QueuedRequest<string, int>("test1"),
            new QueuedRequest<string, int>("test2"),
            new QueuedRequest<string, int>("test3"),
            new QueuedRequest<string, int>("test4")
        };
        var exception = new TimeoutException("Processing timeout");

        scatterer.Scatter(requests, null, exception);

        // All requests should be faulted with the same exception
        Assert.IsTrue(requests[0].ResponseSource.Task.IsFaulted);
        Assert.IsTrue(requests[1].ResponseSource.Task.IsFaulted);
        Assert.IsTrue(requests[2].ResponseSource.Task.IsFaulted);
        Assert.IsTrue(requests[3].ResponseSource.Task.IsFaulted);
    }

    [TestMethod]
    public void ScatterWithPartialFailures_OnlyExceptionSet_StillFails()
    {
        var scatterer = new ResponseScatterer<int>();
        var requests = new[]
        {
            new QueuedRequest<string, int>("test1")
        };
        var responses = new List<int> { 0 };
        var exceptions = new List<Exception> { new Exception("Error") };

        scatterer.ScatterWithPartialFailures(requests, responses, exceptions);

        Assert.IsTrue(requests[0].ResponseSource.Task.IsFaulted);
    }
}
