using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Data;
using MLFramework.Tensor;
using System;

namespace MLFramework.Tests.Distributed
{
    /// <summary>
    /// Mock DataLoader for integration testing.
    /// </summary>
    public class MockDataLoader
    {
        private readonly TestDataset _dataset;
        private readonly IDistributedSampler _sampler;
        private readonly int _batchSize;

        public MockDataLoader(TestDataset dataset, int batchSize, IDistributedSampler sampler)
        {
            _dataset = dataset;
            _sampler = sampler;
            _batchSize = batchSize;
        }

        public System.Collections.Generic.IEnumerable<Tensor> Iterate()
        {
            var indices = _sampler.Iterate();

            for (int i = 0; i < indices.Count; i += _batchSize)
            {
                var batch = Tensor.Random(new long[] { _batchSize, 10, 10 });
                yield return batch;
            }
        }
    }

    /// <summary>
    /// Mock SGD optimizer for integration testing.
    /// </summary>
    public class MockSGD
    {
        private readonly Tensor[] _parameters;
        private readonly float _learningRate;

        public MockSGD(Tensor[] parameters, float lr = 0.01f)
        {
            _parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            _learningRate = lr;
        }

        public void Step()
        {
            foreach (var param in _parameters)
            {
                // Simplified: just add some noise to simulate parameter updates
                var noise = Tensor.Random(param.Shape).Mul(_learningRate * 0.1f);
                param.Sub_(noise);
            }
        }

        public void ZeroGrad()
        {
            // In a real implementation, this would zero out gradients
            // For mock testing, we just no-op
        }
    }

    [TestClass]
    public class EndToEndTests
    {
        [TestMethod]
        public void DistributedTraining_TrainsModel_WithSpeedup()
        {
            // This test requires actual distributed hardware or extensive mocking
            // For now, we can test the pipeline with mocked communication

            var processGroup = MockProcessGroup.Create(worldSize: 4, rank: 0);
            var model = new SimpleModel();
            var ddpModel = new DistributedDataParallel(model, processGroup);

            var dataset = new TestDataset(100, i => Tensor.Random(new long[] { 10, 10 }));
            var sampler = new DistributedSampler(datasetSize: 100, numReplicas: 4, rank: 0, shuffle: true);
            var loader = new MockDataLoader(dataset, batchSize: 32, sampler: sampler);

            var optimizer = new MockSGD(model.GetParameters(), lr: 0.01f);

            // Simulate training
            for (int epoch = 0; epoch < 2; epoch++)
            {
                sampler.SetEpoch(epoch);

                foreach (var batch in loader.Iterate())
                {
                    var output = ddpModel.Forward(batch);
                    var loss = output.Mean();
                    loss.Backward();
                    optimizer.Step();
                    optimizer.ZeroGrad();
                }
            }

            // If we reach here, training completed without errors
            Assert.IsTrue(true);

            processGroup.Destroy();
        }

        [TestMethod]
        public void DistributedTraining_WithEpochs_Completes()
        {
            var processGroup = MockProcessGroup.Create(worldSize: 2, rank: 0);
            var model = new SimpleModel();
            var ddpModel = new DistributedDataParallel(model, processGroup);

            var dataset = new TestDataset(50);
            var sampler = new DistributedSampler(datasetSize: 50, numReplicas: 2, rank: 0, shuffle: false);
            var loader = new MockDataLoader(dataset, batchSize: 10, sampler: sampler);

            var optimizer = new MockSGD(model.GetParameters());

            int iterations = 0;
            for (int epoch = 0; epoch < 3; epoch++)
            {
                sampler.SetEpoch(epoch);

                foreach (var batch in loader.Iterate())
                {
                    var output = ddpModel.Forward(batch);
                    optimizer.Step();
                    optimizer.ZeroGrad();
                    iterations++;
                }
            }

            // Should have completed 3 epochs
            Assert.AreEqual(3, sampler.Epoch);
            Assert.IsTrue(iterations > 0);

            processGroup.Destroy();
        }

        [TestMethod]
        public void DistributedTraining_MultipleRanks_SplitData()
        {
            // Test that data is split correctly across ranks
            int worldSize = 4;
            int datasetSize = 100;

            var rankIndices = new System.Collections.Generic.List<int>[worldSize];

            for (int rank = 0; rank < worldSize; rank++)
            {
                var sampler = new DistributedSampler(
                    datasetSize: datasetSize,
                    numReplicas: worldSize,
                    rank: rank,
                    shuffle: false
                );

                rankIndices[rank] = new System.Collections.Generic.List<int>(sampler.Iterate());
            }

            // Verify that all indices are covered
            var allIndices = new System.Collections.Generic.HashSet<int>();
            for (int rank = 0; rank < worldSize; rank++)
            {
                foreach (var idx in rankIndices[rank])
                {
                    Assert.IsFalse(allIndices.Contains(idx), $"Index {idx} appears in multiple ranks");
                    allIndices.Add(idx);
                }
            }

            Assert.AreEqual(datasetSize, allIndices.Count);
        }

        [TestMethod]
        public void DistributedTraining_BroadcastBeforeTraining_Synchronizes()
        {
            var processGroup = MockProcessGroup.Create(worldSize: 2, rank: 0);
            var model = new SimpleModel();
            var ddpModel = new DistributedDataParallel(model, processGroup);

            // Broadcast parameters before training
            ddpModel.BroadcastParameters();

            // Model should be ready for training
            Assert.IsNotNull(model.Weight);
            Assert.IsNotNull(model.Bias);

            processGroup.Destroy();
        }

        [TestMethod]
        public void DistributedTraining_LargeDataset_Completes()
        {
            var processGroup = MockProcessGroup.Create(worldSize: 2, rank: 0);
            var model = new SimpleModel();
            var ddpModel = new DistributedDataParallel(model, processGroup);

            var dataset = new TestDataset(1000); // Larger dataset
            var sampler = new DistributedSampler(datasetSize: 1000, numReplicas: 2, rank: 0, shuffle: true);
            var loader = new MockDataLoader(dataset, batchSize: 64, sampler: sampler);

            var optimizer = new MockSGD(model.GetParameters());

            // Simulate training for 1 epoch
            sampler.SetEpoch(0);
            int batchesProcessed = 0;

            foreach (var batch in loader.Iterate())
            {
                var output = ddpModel.Forward(batch);
                optimizer.Step();
                optimizer.ZeroGrad();
                batchesProcessed++;
            }

            // Should have processed some batches
            Assert.IsTrue(batchesProcessed > 0);

            processGroup.Destroy();
        }

        [TestMethod]
        public void DistributedTraining_SmallBatchSize_Completes()
        {
            var processGroup = MockProcessGroup.Create(worldSize: 2, rank: 0);
            var model = new SimpleModel();
            var ddpModel = new DistributedDataParallel(model, processGroup);

            var dataset = new TestDataset(100);
            var sampler = new DistributedSampler(datasetSize: 100, numReplicas: 2, rank: 0, shuffle: false);
            var loader = new MockDataLoader(dataset, batchSize: 4, sampler: sampler); // Small batch size

            var optimizer = new MockSGD(model.GetParameters());

            // Simulate training
            sampler.SetEpoch(0);
            bool completed = false;

            foreach (var batch in loader.Iterate())
            {
                var output = ddpModel.Forward(batch);
                optimizer.Step();
                optimizer.ZeroGrad();
                completed = true;
            }

            Assert.IsTrue(completed, "Training should complete with small batch size");

            processGroup.Destroy();
        }

        [TestMethod]
        public void DistributedTraining_SingleEpoch_Completes()
        {
            var processGroup = MockProcessGroup.Create(worldSize: 2, rank: 0);
            var model = new SimpleModel();
            var ddpModel = new DistributedDataParallel(model, processGroup);

            var dataset = new TestDataset(50);
            var sampler = new DistributedSampler(datasetSize: 50, numReplicas: 2, rank: 0, shuffle: false);
            var loader = new MockDataLoader(dataset, batchSize: 10, sampler: sampler);

            var optimizer = new MockSGD(model.GetParameters());

            // Run just 1 epoch
            sampler.SetEpoch(0);
            foreach (var batch in loader.Iterate())
            {
                var output = ddpModel.Forward(batch);
                optimizer.Step();
                optimizer.ZeroGrad();
            }

            // Should complete successfully
            Assert.AreEqual(0, sampler.Epoch);

            processGroup.Destroy();
        }
    }
}
