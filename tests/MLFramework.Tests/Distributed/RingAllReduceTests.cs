using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Tensor;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace MLFramework.Tests.Distributed
{
    [TestClass]
    public class RingAllReduceTests
    {
        private MockProcessGroup _processGroup;

        [TestInitialize]
        public void Setup()
        {
            _processGroup = MockProcessGroup.Create(worldSize: 4, rank: 0);
        }

        [TestCleanup]
        public void Cleanup()
        {
            _processGroup?.Destroy();
        }

        [TestMethod]
        public void RingAllReduce_Sum_CorrectlyAggregatesGradients()
        {
            // Simulate 4 ranks with different gradients
            var gradients = new[]
            {
                Tensor.Ones(new long[] { 10 }).Mul(1),    // Rank 0
                Tensor.Ones(new long[] { 10 }).Mul(2),  // Rank 1
                Tensor.Ones(new long[] { 10 }).Mul(3),  // Rank 2
                Tensor.Ones(new long[] { 10 }).Mul(4)   // Rank 3
            };

            // Expected sum: 1 + 2 + 3 + 4 = 10
            var expected = Tensor.Ones(new long[] { 10 }).Mul(10);

            _processGroup.SimulateAllReduce(new List<Tensor>(gradients), ReduceOp.Sum);

            for (int i = 0; i < gradients.Length; i++)
            {
                Assert.IsTrue(Tensor.AllClose(gradients[i], expected), $"Gradient {i} does not match expected sum");
            }
        }

        [TestMethod]
        public void RingAllReduce_Avg_CorrectlyComputesAverage()
        {
            var gradients = new[]
            {
                Tensor.Ones(new long[] { 10 }).Mul(1),
                Tensor.Ones(new long[] { 10 }).Mul(3),
                Tensor.Ones(new long[] { 10 }).Mul(5)
            };

            // Expected avg: (1 + 3 + 5) / 3 = 3
            var expected = Tensor.Ones(new long[] { 10 }).Mul(3);

            _processGroup.SimulateAllReduce(new List<Tensor>(gradients), ReduceOp.Avg);

            foreach (var grad in gradients)
            {
                Assert.IsTrue(Tensor.AllClose(grad, expected), "Gradient does not match expected average");
            }
        }

        [TestMethod]
        public void RingAllReduce_Max_CorrectlyFindsMaximum()
        {
            var gradients = new[]
            {
                Tensor.Random(new long[] { 10 }).Mul(10),
                Tensor.Random(new long[] { 10 }).Mul(10),
                Tensor.Random(new long[] { 10 }).Mul(10)
            };

            var expected = Tensor.Maximum(Tensor.Maximum(gradients[0], gradients[1]), gradients[2]);

            _processGroup.SimulateAllReduce(new List<Tensor>(gradients), ReduceOp.Max);

            foreach (var grad in gradients)
            {
                Assert.IsTrue(Tensor.AllClose(grad, expected), "Gradient does not match expected maximum");
            }
        }

        [TestMethod]
        public void RingAllReduce_AllReduceAsync_CompletesSuccessfully()
        {
            var tensor = Tensor.Random(new long[] { 1000 });

            var task = _processGroup.AllReduceAsync(tensor, ReduceOp.Sum);
            Assert.IsTrue(task.IsCompleted, "Async operation should complete immediately in mock");
        }

        [TestMethod]
        public void RingAllReduce_SingleDevice_SkipsCommunication()
        {
            _processGroup?.Destroy(); // Clean up first group

            var singleDeviceGroup = MockProcessGroup.Create(worldSize: 1, rank: 0);
            var tensor = Tensor.Random(new long[] { 10 });

            var original = tensor.Clone();
            singleDeviceGroup.AllReduce(tensor, ReduceOp.Sum);

            // Tensor should be unchanged (single device)
            Assert.IsTrue(Tensor.AllClose(tensor, original), "Tensor should be unchanged for single device");

            singleDeviceGroup.Destroy();
        }

        [TestMethod]
        public void RingAllReduce_LargeTensor_CorrectlyAggregates()
        {
            // Test with larger tensors
            var gradients = new[]
            {
                Tensor.Ones(new long[] { 100, 100 }).Mul(1),
                Tensor.Ones(new long[] { 100, 100 }).Mul(2),
                Tensor.Ones(new long[] { 100, 100 }).Mul(3)
            };

            var expected = Tensor.Ones(new long[] { 100, 100 }).Mul(6);

            _processGroup.SimulateAllReduce(new List<Tensor>(gradients), ReduceOp.Sum);

            foreach (var grad in gradients)
            {
                Assert.IsTrue(Tensor.AllClose(grad, expected), "Large tensor gradient aggregation failed");
            }
        }

        [TestMethod]
        public void RingAllReduce_ZeroValues_HandledCorrectly()
        {
            var gradients = new[]
            {
                Tensor.Zeros(new long[] { 5 }),
                Tensor.Ones(new long[] { 5 }).Mul(5),
                Tensor.Ones(new long[] { 5 }).Mul(10)
            };

            var expected = Tensor.Ones(new long[] { 5 }).Mul(15);

            _processGroup.SimulateAllReduce(new List<Tensor>(gradients), ReduceOp.Sum);

            foreach (var grad in gradients)
            {
                Assert.IsTrue(Tensor.AllClose(grad, expected), "Zero values not handled correctly");
            }
        }

        [TestMethod]
        public void RingAllReduce_NegativeValues_CorrectlyAggregated()
        {
            var gradients = new[]
            {
                Tensor.Ones(new long[] { 5 }).Mul(-5),
                Tensor.Ones(new long[] { 5 }).Mul(10),
                Tensor.Ones(new long[] { 5 }).Mul(-2)
            };

            var expected = Tensor.Ones(new long[] { 5 }).Mul(3); // -5 + 10 + -2 = 3

            _processGroup.SimulateAllReduce(new List<Tensor>(gradients), ReduceOp.Sum);

            foreach (var grad in gradients)
            {
                Assert.IsTrue(Tensor.AllClose(grad, expected), "Negative values not aggregated correctly");
            }
        }
    }
}
