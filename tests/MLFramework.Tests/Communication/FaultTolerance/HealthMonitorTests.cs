namespace MLFramework.Tests.Communication.FaultTolerance;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Communication;
using MLFramework.Communication.FaultTolerance;
using MLFramework.Communication.Tests;
using System.Threading.Tasks;

[TestClass]
public class HealthMonitorTests
{
    private MockCommunicationBackend _mockBackend;

    [TestInitialize]
    public void Setup()
    {
        _mockBackend = new MockCommunicationBackend(0, 4);
    }

    [TestCleanup]
    public void Cleanup()
    {
        _mockBackend?.Dispose();
    }

    [TestMethod]
    public void TestHealthMonitor_Constructor_InitializesCorrectly()
    {
        var monitor = new HealthMonitor(_mockBackend);

        Assert.AreEqual(0, monitor.UnresponsiveRanksCount);
        Assert.AreEqual(RankHealthStatus.Healthy, monitor.GetRankHealthStatus(0));
        Assert.AreEqual(RankHealthStatus.Healthy, monitor.GetRankHealthStatus(1));
        Assert.AreEqual(RankHealthStatus.Healthy, monitor.GetRankHealthStatus(2));
        Assert.AreEqual(RankHealthStatus.Healthy, monitor.GetRankHealthStatus(3));

        monitor.Dispose();
    }

    [TestMethod]
    public void TestHealthMonitor_CustomTimeout_InitializesCorrectly()
    {
        var monitor = new HealthMonitor(_mockBackend, TimeSpan.FromSeconds(10));

        Assert.AreEqual(0, monitor.UnresponsiveRanksCount);

        monitor.Dispose();
    }

    [TestMethod]
    public void TestHealthMonitor_UpdateHeartbeat_UpdatesRankStatus()
    {
        var monitor = new HealthMonitor(_mockBackend);

        monitor.UpdateHeartbeat(1);
        Assert.AreEqual(RankHealthStatus.Healthy, monitor.GetRankHealthStatus(1));

        monitor.Dispose();
    }

    [TestMethod]
    public void TestHealthMonitor_GetRankHealthStatus_ReturnsCorrectStatus()
    {
        var monitor = new HealthMonitor(_mockBackend);

        // All ranks should be healthy initially
        for (int i = 0; i < 4; i++)
        {
            Assert.AreEqual(RankHealthStatus.Healthy, monitor.GetRankHealthStatus(i));
        }

        // Invalid rank should return Failed
        Assert.AreEqual(RankHealthStatus.Failed, monitor.GetRankHealthStatus(99));

        monitor.Dispose();
    }

    [TestMethod]
    public void TestHealthMonitor_GetHealthyRanks_ReturnsAllRanks()
    {
        var monitor = new HealthMonitor(_mockBackend);

        var healthyRanks = monitor.GetHealthyRanks();

        Assert.AreEqual(4, healthyRanks.Count);
        CollectionAssert.AreEquivalent(new[] { 0, 1, 2, 3 }, healthyRanks);

        monitor.Dispose();
    }

    [TestMethod]
    public void TestHealthMonitor_StartAndStopMonitoring_WorksCorrectly()
    {
        var monitor = new HealthMonitor(_mockBackend, TimeSpan.FromSeconds(5));

        monitor.StartMonitoring();

        // Wait a bit
        Task.Delay(100).Wait();

        monitor.StopMonitoring();

        // Should not throw when stopping again
        monitor.StopMonitoring();

        monitor.Dispose();
    }

    [TestMethod]
    public void TestHealthMonitor_UnresponsiveRanksCount_TracksCorrectly()
    {
        var monitor = new HealthMonitor(_mockBackend);

        // Initially, all ranks are healthy
        Assert.AreEqual(0, monitor.UnresponsiveRanksCount);

        // Update heartbeat for some ranks (others will become unresponsive over time)
        monitor.UpdateHeartbeat(0);
        monitor.UpdateHeartbeat(1);

        // Still all healthy since they were just updated
        Assert.AreEqual(0, monitor.UnresponsiveRanksCount);

        monitor.Dispose();
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void TestHealthMonitor_NullBackend_ThrowsException()
    {
        var monitor = new HealthMonitor(null!);
    }

    [TestMethod]
    public void TestHealthMonitor_UpdateHeartbeat_InvalidRank_DoesNotThrow()
    {
        var monitor = new HealthMonitor(_mockBackend);

        // Should not throw for invalid rank
        monitor.UpdateHeartbeat(99);

        monitor.Dispose();
    }

    [TestMethod]
    public void TestHealthMonitor_MultipleDispose_DoesNotThrow()
    {
        var monitor = new HealthMonitor(_mockBackend);

        monitor.Dispose();
        monitor.Dispose(); // Should not throw
    }

    [TestMethod]
    public async Task TestHealthMonitor_Monitoring_TaskCompletes()
    {
        var monitor = new HealthMonitor(_mockBackend, TimeSpan.FromSeconds(1));

        monitor.StartMonitoring();

        // Wait for a few health check cycles
        await Task.Delay(600);

        monitor.StopMonitoring();
        monitor.Dispose();
    }

    [TestMethod]
    public void TestHealthMonitor_GetHealthyRanks_WhenAllHealthy_ReturnsAll()
    {
        var monitor = new HealthMonitor(_mockBackend);

        var healthyRanks = monitor.GetHealthyRanks();

        Assert.AreEqual(4, healthyRanks.Count);
        Assert.IsTrue(healthyRanks.Contains(0));
        Assert.IsTrue(healthyRanks.Contains(1));
        Assert.IsTrue(healthyRanks.Contains(2));
        Assert.IsTrue(healthyRanks.Contains(3));

        monitor.Dispose();
    }
}
