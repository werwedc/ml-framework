namespace MLFramework.Tests.Communication.FaultTolerance;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Communication;
using MLFramework.Communication.FaultTolerance;
using MLFramework.Communication.Tests;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;

[TestClass]
public class FaultTolerantCommunicationTests
{
    private MockCommunicationBackend _mockBackend;
    private CommunicationConfig _config;

    [TestInitialize]
    public void Setup()
    {
        _mockBackend = new MockCommunicationBackend(0, 2);
        _config = new CommunicationConfig
        {
            TimeoutMs = 5000,
            MaxRetries = 3,
            RetryDelayMs = 100
        };
    }

    [TestCleanup]
    public void Cleanup()
    {
        _mockBackend?.Dispose();
    }

    [TestMethod]
    public void TestFaultTolerantCommunication_Constructor_InitializesCorrectly()
    {
        var ftComm = new FaultTolerantCommunication(_mockBackend, _config);
        Assert.AreEqual(0, ftComm.Rank);
        Assert.AreEqual(2, ftComm.WorldSize);
        Assert.IsTrue(ftComm.BackendName.StartsWith("FT_"));
        ftComm.Dispose();
    }

    [TestMethod]
    public void TestFaultTolerantCommunication_Broadcast_WrapsBackend()
    {
        var ftComm = new FaultTolerantCommunication(_mockBackend, _config);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });

        ftComm.Broadcast(tensor, 0);

        ftComm.Dispose();
    }

    [TestMethod]
    public void TestFaultTolerantCommunication_AllReduce_WrapsBackend()
    {
        var ftComm = new FaultTolerantCommunication(_mockBackend, _config);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });

        var result = ftComm.AllReduce(tensor, ReduceOp.Sum);

        Assert.IsNotNull(result);
        ftComm.Dispose();
    }

    [TestMethod]
    public void TestFaultTolerantCommunication_Reduce_WrapsBackend()
    {
        var ftComm = new FaultTolerantCommunication(_mockBackend, _config);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });

        var result = ftComm.Reduce(tensor, ReduceOp.Sum, 0);

        Assert.IsNotNull(result);
        ftComm.Dispose();
    }

    [TestMethod]
    public void TestFaultTolerantCommunication_AllGather_WrapsBackend()
    {
        var ftComm = new FaultTolerantCommunication(_mockBackend, _config);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });

        var result = ftComm.AllGather(tensor);

        Assert.IsNotNull(result);
        ftComm.Dispose();
    }

    [TestMethod]
    public void TestFaultTolerantCommunication_ReduceScatter_WrapsBackend()
    {
        var ftComm = new FaultTolerantCommunication(_mockBackend, _config);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });

        var result = ftComm.ReduceScatter(tensor, ReduceOp.Sum);

        Assert.IsNotNull(result);
        ftComm.Dispose();
    }

    [TestMethod]
    public void TestFaultTolerantCommunication_Barrier_WrapsBackend()
    {
        var ftComm = new FaultTolerantCommunication(_mockBackend, _config);

        ftComm.Barrier();

        ftComm.Dispose();
    }

    [TestMethod]
    public void TestFaultTolerantCommunication_GetErrorRecoveryManager_ReturnsManager()
    {
        var ftComm = new FaultTolerantCommunication(_mockBackend, _config);

        var manager = ftComm.ErrorRecoveryManager;
        Assert.IsNotNull(manager);
        Assert.AreEqual(3, manager.MaxRetries);

        ftComm.Dispose();
    }

    [TestMethod]
    public void TestFaultTolerantCommunication_GetTimeoutManager_ReturnsManager()
    {
        var ftComm = new FaultTolerantCommunication(_mockBackend, _config);

        var manager = ftComm.TimeoutManager;
        Assert.IsNotNull(manager);
        Assert.AreEqual(5000, manager.DefaultTimeoutMs);

        ftComm.Dispose();
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void TestFaultTolerantCommunication_NullBackend_ThrowsException()
    {
        var ftComm = new FaultTolerantCommunication(null!, _config);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void TestFaultTolerantCommunication_NullConfig_ThrowsException()
    {
        var ftComm = new FaultTolerantCommunication(_mockBackend, null!);
    }

    [TestMethod]
    public void TestFaultTolerantCommunication_Dispose_DisposesManagers()
    {
        var ftComm = new FaultTolerantCommunication(_mockBackend, _config);

        ftComm.Dispose();

        // Should not throw when disposing again
        ftComm.Dispose();
    }
}
