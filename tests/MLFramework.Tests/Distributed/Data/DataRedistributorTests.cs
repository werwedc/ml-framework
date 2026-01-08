using Microsoft.VisualStudio.TestTools.UnitTesting;
using MachineLearning.Distributed.Configuration;
using MachineLearning.Distributed.Data;
using MachineLearning.Distributed.Enums;
using MachineLearning.Distributed.Models;
using System.Collections.Generic;

namespace MLFramework.Tests.Distributed.Data;

[TestClass]
public class DataRedistributorTests
{
    private ElasticTrainingConfig _config = null!;
    private DataRedistributor _redistributor = null!;

    [TestInitialize]
    public void Setup()
    {
        _config = new ElasticTrainingConfig
        {
            MinWorkers = 1,
            MaxWorkers = 10,
            RescaleTimeoutMs = 1000,
            StabilityWindowMs = 500,
            WorkerHeartbeatTimeoutMs = 2000,
            RedistributionType = RedistributionType.FullReshuffle
        };

        _redistributor = new DataRedistributor(_config);
    }

    [TestMethod]
    public void Constructor_ValidConfig_CreatesRedistributor()
    {
        // Assert
        Assert.IsNotNull(_redistributor);
    }

    [TestMethod]
    public void CalculatePlan_FullReshuffle_RedistributesAllShards()
    {
        // Arrange
        var oldTopology = new ClusterTopology
        {
            WorldSize = 2,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001)
            }
        };

        var newTopology = new ClusterTopology
        {
            WorldSize = 4,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001),
                new WorkerId("worker2", "localhost", 8002),
                new WorkerId("worker3", "localhost", 8003)
            }
        };

        // Act
        var plan = _redistributor.CalculatePlan(oldTopology, newTopology, 100);

        // Assert
        Assert.IsTrue(plan.Transfers.Count > 0);
        Assert.AreEqual(100, plan.TotalShards);
        Assert.AreEqual(4, plan.WorkerAssignments.Count);
    }

    [TestMethod]
    public void CalculatePlan_Incremental_RedistributesOnlyNecessaryShards()
    {
        // Arrange
        _config.RedistributionType = RedistributionType.Incremental;
        var redistributor = new DataRedistributor(_config);

        var oldTopology = new ClusterTopology
        {
            WorldSize = 2,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001)
            }
        };

        var newTopology = new ClusterTopology
        {
            WorldSize = 4,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001),
                new WorkerId("worker2", "localhost", 8002),
                new WorkerId("worker3", "localhost", 8003)
            }
        };

        // Act
        var plan = redistributor.CalculatePlan(oldTopology, newTopology, 100);

        // Assert
        Assert.IsTrue(plan.Transfers.Count > 0);
        Assert.AreEqual(100, plan.TotalShards);
        // Incremental should have fewer transfers than full reshuffle
        Assert.IsTrue(plan.Transfers.Count < 100);
    }

    [TestMethod]
    public void CalculatePlan_SameTopology_NoTransfers()
    {
        // Arrange
        var topology = new ClusterTopology
        {
            WorldSize = 2,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001)
            }
        };

        // Act
        var plan = _redistributor.CalculatePlan(topology, topology, 100);

        // Assert
        Assert.AreEqual(0, plan.Transfers.Count);
        Assert.AreEqual(100, plan.TotalShards);
    }

    [TestMethod]
    public void CalculatePlan_ScaleDown_CreatesCorrectPlan()
    {
        // Arrange
        var oldTopology = new ClusterTopology
        {
            WorldSize = 4,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001),
                new WorkerId("worker2", "localhost", 8002),
                new WorkerId("worker3", "localhost", 8003)
            }
        };

        var newTopology = new ClusterTopology
        {
            WorldSize = 2,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001)
            }
        };

        // Act
        var plan = _redistributor.CalculatePlan(oldTopology, newTopology, 100);

        // Assert
        Assert.IsTrue(plan.Transfers.Count > 0);
        Assert.AreEqual(2, plan.WorkerAssignments.Count);
    }

    [TestMethod]
    public void ValidateRedistribution_CorrectPlan_ReturnsTrue()
    {
        // Arrange
        var topology = new ClusterTopology
        {
            WorldSize = 4,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001),
                new WorkerId("worker2", "localhost", 8002),
                new WorkerId("worker3", "localhost", 8003)
            }
        };

        var plan = _redistributor.CalculatePlan(topology, topology, 100);

        // Act
        var isValid = _redistributor.ValidateRedistribution(topology, plan);

        // Assert
        Assert.IsTrue(isValid);
    }

    [TestMethod]
    public void ValidateRedistribution_IncorrectPlan_ReturnsFalse()
    {
        // Arrange
        var topology = new ClusterTopology
        {
            WorldSize = 2,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001)
            }
        };

        var plan = new DataRedistributionPlan
        {
            TotalShards = 100,
            Transfers = new List<DataTransfer>(),
            WorkerAssignments = new Dictionary<WorkerId, List<DataShard>>
            {
                { topology.Workers[0], new List<DataShard> { new DataShard(0, 0, 25) } }
            }
        };

        // Act
        var isValid = _redistributor.ValidateRedistribution(topology, plan);

        // Assert
        Assert.IsFalse(isValid);
    }

    [TestMethod]
    public void TransferDataShardsAsync_CompletesSuccessfully()
    {
        // Arrange
        var source = new WorkerId("worker0", "localhost", 8000);
        var destination = new WorkerId("worker1", "localhost", 8001);
        var shard = new DataShard(0, 0, 100);

        // Act & Assert
        // Should not throw
        Assert.IsNotNull(_redistributor.TransferDataShardsAsync(source, destination, shard));
    }

    [TestMethod]
    public async Task TransferDataShardsAsync_IsAwaitable()
    {
        // Arrange
        var source = new WorkerId("worker0", "localhost", 8000);
        var destination = new WorkerId("worker1", "localhost", 8001);
        var shard = new DataShard(0, 0, 100);

        // Act & Assert
        await _redistributor.TransferDataShardsAsync(source, destination, shard);
    }

    [TestMethod]
    public void CalculatePlan_WorkerAssignments_DistributedCorrectly()
    {
        // Arrange
        var oldTopology = new ClusterTopology
        {
            WorldSize = 2,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001)
            }
        };

        var newTopology = new ClusterTopology
        {
            WorldSize = 4,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001),
                new WorkerId("worker2", "localhost", 8002),
                new WorkerId("worker3", "localhost", 8003)
            }
        };

        // Act
        var plan = _redistributor.CalculatePlan(oldTopology, newTopology, 100);

        // Assert - All workers should have assignments
        Assert.IsTrue(plan.WorkerAssignments.ContainsKey(newTopology.Workers[0]));
        Assert.IsTrue(plan.WorkerAssignments.ContainsKey(newTopology.Workers[1]));
        Assert.IsTrue(plan.WorkerAssignments.ContainsKey(newTopology.Workers[2]));
        Assert.IsTrue(plan.WorkerAssignments.ContainsKey(newTopology.Workers[3]));
    }

    [TestMethod]
    public void CalculatePlan_TransferPriorities_DecreaseWithIndex()
    {
        // Arrange
        var oldTopology = new ClusterTopology
        {
            WorldSize = 2,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001)
            }
        };

        var newTopology = new ClusterTopology
        {
            WorldSize = 4,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001),
                new WorkerId("worker2", "localhost", 8002),
                new WorkerId("worker3", "localhost", 8003)
            }
        };

        // Act
        var plan = _redistributor.CalculatePlan(oldTopology, newTopology, 100);

        // Assert - Earlier shards should have higher priority
        if (plan.Transfers.Count >= 2)
        {
            var firstTransfer = plan.Transfers[0];
            var lastTransfer = plan.Transfers[plan.Transfers.Count - 1];
            Assert.IsTrue(firstTransfer.Priority > lastTransfer.Priority);
        }
    }

    [TestMethod]
    public void CalculatePlan_EvenDistribution_AllWorkersGetEqualShards()
    {
        // Arrange
        var topology = new ClusterTopology
        {
            WorldSize = 4,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001),
                new WorkerId("worker2", "localhost", 8002),
                new WorkerId("worker3", "localhost", 8003)
            }
        };

        // Act
        var plan = _redistributor.CalculatePlan(topology, topology, 100);

        // Assert
        foreach (var assignment in plan.WorkerAssignments)
        {
            Assert.AreEqual(25, assignment.Value.Count);
        }
    }

    [TestMethod]
    public void CalculatePlan_UnevenDatasetSize_DistributesRemainder()
    {
        // Arrange
        var topology = new ClusterTopology
        {
            WorldSize = 4,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001),
                new WorkerId("worker2", "localhost", 8002),
                new WorkerId("worker3", "localhost", 8003)
            }
        };

        // Act
        var plan = _redistributor.CalculatePlan(topology, topology, 103);

        // Assert - First 3 workers get 26, last gets 25
        Assert.AreEqual(26, plan.WorkerAssignments[topology.Workers[0]].Count);
        Assert.AreEqual(26, plan.WorkerAssignments[topology.Workers[1]].Count);
        Assert.AreEqual(26, plan.WorkerAssignments[topology.Workers[2]].Count);
        Assert.AreEqual(25, plan.WorkerAssignments[topology.Workers[3]].Count);
    }
}

[TestClass]
public class DataTransferManagerTests
{
    private DataTransferManager _manager = null!;

    [TestInitialize]
    public void Setup()
    {
        _manager = new DataTransferManager(4);
    }

    [TestMethod]
    public void Constructor_DefaultConcurrency_CreatesManager()
    {
        // Act
        var manager = new DataTransferManager();

        // Assert
        Assert.IsNotNull(manager);
    }

    [TestMethod]
    public void Constructor_CustomConcurrency_CreatesManager()
    {
        // Act
        var manager = new DataTransferManager(8);

        // Assert
        Assert.IsNotNull(manager);
    }

    [TestMethod]
    public void Constructor_ZeroConcurrency_ThrowsException()
    {
        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() =>
            new DataTransferManager(0));
    }

    [TestMethod]
    public void Constructor_NegativeConcurrency_ThrowsException()
    {
        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() =>
            new DataTransferManager(-1));
    }

    [TestMethod]
    public async Task ExecutePlanAsync_EmptyPlan_CompletesSuccessfully()
    {
        // Arrange
        var plan = new DataRedistributionPlan
        {
            TotalShards = 0,
            Transfers = new List<DataTransfer>(),
            WorkerAssignments = new Dictionary<WorkerId, List<DataShard>>()
        };

        // Act & Assert
        await _manager.ExecutePlanAsync(plan);
    }

    [TestMethod]
    public async Task ExecutePlanAsync_PlanWithTransfers_CompletesSuccessfully()
    {
        // Arrange
        var plan = new DataRedistributionPlan
        {
            TotalShards = 10,
            Transfers = new List<DataTransfer>
            {
                new DataTransfer
                {
                    SourceWorker = new WorkerId("worker0", "localhost", 8000),
                    DestinationWorker = new WorkerId("worker1", "localhost", 8001),
                    Shard = new DataShard(0, 0, 10),
                    Priority = 10,
                    EstimatedCompletionTime = DateTime.UtcNow
                }
            },
            WorkerAssignments = new Dictionary<WorkerId, List<DataShard>>()
        };

        // Act & Assert
        await _manager.ExecutePlanAsync(plan);
    }

    [TestMethod]
    public async Task ExecutePlanAsync_WithCancellation_CancelsExecution()
    {
        // Arrange
        var plan = new DataRedistributionPlan
        {
            TotalShards = 10,
            Transfers = new List<DataTransfer>
            {
                new DataTransfer
                {
                    SourceWorker = new WorkerId("worker0", "localhost", 8000),
                    DestinationWorker = new WorkerId("worker1", "localhost", 8001),
                    Shard = new DataShard(0, 0, 10),
                    Priority = 10,
                    EstimatedCompletionTime = DateTime.UtcNow
                }
            },
            WorkerAssignments = new Dictionary<WorkerId, List<DataShard>>()
        };

        var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act & Assert
        await Assert.ThrowsExceptionAsync<TaskCanceledException>(() =>
            _manager.ExecutePlanAsync(plan, cts.Token));
    }
}
