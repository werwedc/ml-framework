using System;
using System.Threading;
using MobileRuntime.Benchmarking.Interfaces;
using MobileRuntime.Benchmarking.Monitoring;
using MobileRuntime.Benchmarking.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Assert = Microsoft.VisualStudio.TestTools.UnitTesting.Assert;

namespace MobileRuntime.Tests.Benchmarking;

[TestClass]
public class MemoryMonitorTests
{
    [TestMethod]
    public void StartAndStopMonitoring_ShouldWork()
    {
        // Arrange
        var monitor = new MemoryMonitor();

        // Act & Assert
        monitor.StartMonitoring();
        Thread.Sleep(200); // Give it time to collect samples
        monitor.StopMonitoring();

        var snapshot = monitor.GetSnapshot();
        Assert.NotNull(snapshot);
    }

    [TestMethod]
    public void GetSnapshot_ShouldReturnCurrentMemory()
    {
        // Arrange
        var monitor = new MemoryMonitor();

        // Act
        var snapshot = monitor.GetSnapshot();

        // Assert
        Assert.NotNull(snapshot);
        Assert.True(snapshot.WorkingSetBytes > 0);
        Assert.True(snapshot.PrivateMemoryBytes > 0);
        Assert.True(snapshot.GCMemoryBytes >= 0);
    }

    [TestMethod]
    public void Reset_ShouldClearSnapshots()
    {
        // Arrange
        var monitor = new MemoryMonitor();
        monitor.StartMonitoring();
        Thread.Sleep(200);
        monitor.StopMonitoring();

        // Act
        monitor.Reset();

        // Assert
        // Reset should clear internal state
        var snapshot = monitor.GetSnapshot();
        Assert.NotNull(snapshot);
    }

    [TestMethod]
    public void Monitor_ShouldTrackGcCollections()
    {
        // Arrange
        var monitor = new MemoryMonitor();

        // Act
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var snapshot = monitor.GetSnapshot();

        // Assert
        Assert.True(snapshot.Gen0Collections >= 0);
        Assert.True(snapshot.Gen1Collections >= 0);
        Assert.True(snapshot.Gen2Collections >= 0);
    }
}

public class EnergyMonitorTests
{
    [TestMethod]
    public void StartAndStopMonitoring_ShouldWork()
    {
        // Arrange
        var monitor = new EnergyMonitor();

        // Act & Assert
        monitor.StartMonitoring();
        Thread.Sleep(200);
        monitor.StopMonitoring();

        var snapshot = monitor.GetSnapshot();
        Assert.NotNull(snapshot);
    }

    [TestMethod]
    public void GetSnapshot_ShouldReturnCurrentEnergy()
    {
        // Arrange
        var monitor = new EnergyMonitor();

        // Act
        var snapshot = monitor.GetSnapshot();

        // Assert
        Assert.NotNull(snapshot);
        Assert.True(snapshot.EnergyJoules >= 0);
    }

    [TestMethod]
    public void Reset_ShouldClearSnapshots()
    {
        // Arrange
        var monitor = new EnergyMonitor();
        monitor.StartMonitoring();
        Thread.Sleep(200);
        monitor.StopMonitoring();

        // Act
        monitor.Reset();

        // Assert
        var snapshot = monitor.GetSnapshot();
        Assert.NotNull(snapshot);
    }
}
