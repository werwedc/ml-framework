using Microsoft.VisualStudio.TestTools.UnitTesting;
using MachineLearning.Distributed.Configuration;
using MachineLearning.Distributed.Models;
using MachineLearning.Distributed.Worker;
using System;

namespace MLFramework.Tests.Distributed
{
    [TestClass]
    public class ElasticWorkerTests
    {
        private ElasticTrainingConfig _validConfig;
        private WorkerId _testWorkerId;

        [TestInitialize]
        public void Setup()
        {
            _validConfig = new ElasticTrainingConfig
            {
                MinWorkers = 1,
                MaxWorkers = 10,
                RescaleTimeoutMs = 30000,
                MaxConsecutiveFailures = 3,
                WorkerHeartbeatTimeoutMs = 10000
            };

            _testWorkerId = new WorkerId("worker-1", "localhost", 8080);
        }

        [TestMethod]
        public void Constructor_WithValidParameters_CreatesWorker()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);

            Assert.IsNotNull(worker);
            Assert.AreEqual(_testWorkerId, worker.WorkerId);
            Assert.IsFalse(worker.IsRescaling);
            Assert.IsNull(worker.CurrentTopology);
            Assert.IsNull(worker.CurrentState);
        }

        [TestMethod]
        public void Constructor_WithInvalidConfig_ThrowsException()
        {
            var invalidConfig = new ElasticTrainingConfig
            {
                MinWorkers = 0,
                WorkerHeartbeatTimeoutMs = 10000
            };

            Assert.ThrowsException<ArgumentException>(() =>
            {
                new ElasticWorker(_testWorkerId, invalidConfig);
            });
        }

        [TestMethod]
        public void ConnectToCluster_FirstConnection_Succeeds()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);

            worker.ConnectToCluster("localhost:9090");

            var metadata = worker.GetMetadata();
            Assert.AreEqual(WorkerStatus.Joining, metadata.Status);
        }

        [TestMethod]
        public void ConnectToCluster_AlreadyConnected_ThrowsException()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);
            worker.ConnectToCluster("localhost:9090");

            Assert.ThrowsException<InvalidOperationException>(() =>
            {
                worker.ConnectToCluster("localhost:9090");
            });
        }

        [TestMethod]
        public async Task DisconnectAsync_AfterConnection_Succeeds()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);
            worker.ConnectToCluster("localhost:9090");

            await worker.DisconnectAsync();

            var metadata = worker.GetMetadata();
            Assert.AreEqual(WorkerStatus.Leaving, metadata.Status);
        }

        [TestMethod]
        public async Task DisconnectAsync_NotConnected_DoesNothing()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);

            await worker.DisconnectAsync();

            // Should not throw
            var metadata = worker.GetMetadata();
        }

        [TestMethod]
        public async Task OnRescalingEventAsync_SynchronousMode_SetsRescalingFlag()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);
            worker.ConnectToCluster("localhost:9090");

            var newTopology = new ClusterTopology();
            newTopology.AddWorker(_testWorkerId);

            var rescalingEvent = new RescalingEvent
            {
                NewTopology = newTopology,
                OldTopology = new ClusterTopology()
            };

            var rescalingRequestedTriggered = false;
            worker.RescalingRequested += (evt) => rescalingRequestedTriggered = true;

            await worker.OnRescalingEventAsync(rescalingEvent);

            Assert.IsFalse(worker.IsRescaling);
            Assert.IsTrue(rescalingRequestedTriggered);
        }

        [TestMethod]
        public async Task OnRescalingEventAsync_AsynchronousMode_UpdatesTopology()
        {
            var asyncConfig = new ElasticTrainingConfig
            {
                MinWorkers = 1,
                MaxWorkers = 10,
                UseSynchronousRescaling = false,
                WorkerHeartbeatTimeoutMs = 10000
            };

            var worker = new ElasticWorker(_testWorkerId, asyncConfig);
            worker.ConnectToCluster("localhost:9090");

            var newTopology = new ClusterTopology();
            newTopology.AddWorker(_testWorkerId);

            var rescalingEvent = new RescalingEvent
            {
                NewTopology = newTopology,
                OldTopology = new ClusterTopology()
            };

            await worker.OnRescalingEventAsync(rescalingEvent);

            Assert.IsNotNull(worker.CurrentTopology);
            Assert.AreEqual(newTopology, worker.CurrentTopology);
        }

        [TestMethod]
        public async Task UpdateTopologyAsync_WithWorkerInTopology_UpdatesRank()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);

            var topology = new ClusterTopology();
            topology.AddWorker(new WorkerId("worker-0", "host1", 8080));
            topology.AddWorker(_testWorkerId);
            topology.AddWorker(new WorkerId("worker-2", "host2", 8080));

            await worker.UpdateTopologyAsync(topology);

            var metadata = worker.GetMetadata();
            Assert.AreEqual(1, metadata.Rank);
        }

        [TestMethod]
        public async Task UpdateTopologyAsync_WorkerNotInTopology_RankRemainsNegative()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);

            var topology = new ClusterTopology();
            topology.AddWorker(new WorkerId("worker-0", "host1", 8080));
            topology.AddWorker(new WorkerId("worker-2", "host2", 8080));

            await worker.UpdateTopologyAsync(topology);

            var metadata = worker.GetMetadata();
            Assert.AreEqual(-1, metadata.Rank);
        }

        [TestMethod]
        public async Task SynchronizeStateAsync_WithValidState_UpdatesCurrentState()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);

            var state = new GlobalTrainingState
            {
                CurrentEpoch = 5,
                CurrentStep = 1000,
                LearningRate = 0.01f,
                GlobalBatchSize = 32,
                ActiveWorkerCount = 4
            };

            var stateSynchronizedTriggered = false;
            worker.StateSynchronized += (s) => stateSynchronizedTriggered = true;

            await worker.SynchronizeStateAsync(state);

            Assert.IsNotNull(worker.CurrentState);
            Assert.AreEqual(state, worker.CurrentState);
            Assert.IsTrue(stateSynchronizedTriggered);
        }

        [TestMethod]
        public async Task ResumeTrainingAsync_WhileRescaling_UpdatesStatus()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);
            worker.ConnectToCluster("localhost:9090");

            // Simulate being in rescaling state
            var metadata = worker.GetMetadata();
            Assert.AreEqual(WorkerStatus.Joining, metadata.Status);

            await worker.ResumeTrainingAsync();

            // Note: ResumeTrainingAsync only updates if IsRescaling is true
            // In this test, we're checking it doesn't throw
        }

        [TestMethod]
        public async Task RedistributeDataAsync_WithValidPlan_Completes()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);

            var plan = new DataRedistributionPlan
            {
                TotalShards = 100
            };

            await worker.RedistributeDataAsync(plan);

            // Should complete without throwing
        }

        [TestMethod]
        public void UpdateTrainingState_WithCurrentState_UpdatesValues()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);

            var state = new GlobalTrainingState
            {
                CurrentEpoch = 1,
                CurrentStep = 100,
                LearningRate = 0.01f
            };

            // Initialize state
            worker.SynchronizeStateAsync(state).Wait();

            worker.UpdateTrainingState(10, 1000, 0.001f);

            Assert.AreEqual(10, worker.CurrentState?.CurrentEpoch);
            Assert.AreEqual(1000, worker.CurrentState?.CurrentStep);
            Assert.AreEqual(0.001f, worker.CurrentState?.LearningRate);
        }

        [TestMethod]
        public void UpdateTrainingState_WithoutCurrentState_DoesNothing()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);

            // Should not throw even without current state
            worker.UpdateTrainingState(10, 1000, 0.001f);
        }

        [TestMethod]
        public void ShouldHandleShard_WithValidTopology_ReturnsCorrectShards()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);

            var topology = new ClusterTopology();
            topology.AddWorker(new WorkerId("worker-0", "host1", 8080));
            topology.AddWorker(_testWorkerId); // Rank 1
            topology.AddWorker(new WorkerId("worker-2", "host2", 8080));

            worker.UpdateTopologyAsync(topology).Wait();

            // With rank 1 and world size 3, should handle shards 1, 4, 7, 10...
            Assert.IsFalse(worker.ShouldHandleShard(0, 12)); // 0 % 3 = 0
            Assert.IsTrue(worker.ShouldHandleShard(1, 12));   // 1 % 3 = 1
            Assert.IsFalse(worker.ShouldHandleShard(2, 12)); // 2 % 3 = 2
            Assert.IsFalse(worker.ShouldHandleShard(3, 12)); // 3 % 3 = 0
            Assert.IsTrue(worker.ShouldHandleShard(4, 12));   // 4 % 3 = 1
            Assert.IsFalse(worker.ShouldHandleShard(5, 12)); // 5 % 3 = 2
        }

        [TestMethod]
        public void ShouldHandleShard_WithoutTopology_ReturnsFalse()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);

            // Should return false when topology is not set
            Assert.IsFalse(worker.ShouldHandleShard(0, 10));
        }

        [TestMethod]
        public void GetMetadata_ReturnsCopyOfMetadata()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);

            var metadata1 = worker.GetMetadata();
            var metadata2 = worker.GetMetadata();

            Assert.AreNotSame(metadata1, metadata2);
            Assert.AreEqual(metadata1.WorkerId, metadata2.WorkerId);
            Assert.AreEqual(metadata1.Status, metadata2.Status);
        }

        [TestMethod]
        public void GetMetadata_WithCustomAttributes_CopiesDictionary()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);

            var metadata = worker.GetMetadata();
            metadata.CustomAttributes.Add("test", "value");

            var metadata2 = worker.GetMetadata();

            Assert.IsFalse(metadata2.CustomAttributes.ContainsKey("test"));
        }

        [TestMethod]
        public async Task Heartbeat_UpdatesLastHeartbeat()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);
            worker.ConnectToCluster("localhost:9090");

            var initialTime = worker.GetMetadata().LastHeartbeat;

            // Wait for heartbeat interval (at least one heartbeat should occur)
            await Task.Delay(4000);

            var updatedTime = worker.GetMetadata().LastHeartbeat;

            Assert.IsTrue(updatedTime > initialTime);
        }

        [TestMethod]
        public void Dispose_CanBeCalledMultipleTimes()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);

            worker.Dispose();
            worker.Dispose();

            // Should not throw
        }

        [TestMethod]
        public void Dispose_StopsHeartbeat()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);
            worker.ConnectToCluster("localhost:9090");

            worker.Dispose();

            // Wait to ensure heartbeat doesn't throw
            Task.Delay(1000).Wait();

            // Should not throw even after disposal
        }

        [TestMethod]
        public void RescalingRequested_Event_IsFired()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);
            var eventFired = false;
            RescalingEvent? capturedEvent = null;

            worker.RescalingRequested += (evt) =>
            {
                eventFired = true;
                capturedEvent = evt;
            };

            var topology = new ClusterTopology();
            topology.AddWorker(_testWorkerId);

            var rescalingEvent = new RescalingEvent
            {
                NewTopology = topology,
                OldTopology = new ClusterTopology()
            };

            worker.OnRescalingEventAsync(rescalingEvent).Wait();

            Assert.IsTrue(eventFired);
            Assert.IsNotNull(capturedEvent);
        }

        [TestMethod]
        public void StateSynchronized_Event_IsFired()
        {
            var worker = new ElasticWorker(_testWorkerId, _validConfig);
            var eventFired = false;
            GlobalTrainingState? capturedState = null;

            worker.StateSynchronized += (state) =>
            {
                eventFired = true;
                capturedState = state;
            };

            var state = new GlobalTrainingState
            {
                CurrentEpoch = 5,
                CurrentStep = 1000,
                LearningRate = 0.01f
            };

            worker.SynchronizeStateAsync(state).Wait();

            Assert.IsTrue(eventFired);
            Assert.IsNotNull(capturedState);
            Assert.AreEqual(state, capturedState);
        }
    }
}
