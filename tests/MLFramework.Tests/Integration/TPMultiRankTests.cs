using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Tests.Integration;
using MLFramework.Tensor;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Tests.Integration
{
    [TestClass]
    public class TPMultiRankTests
    {
        [TestMethod]
        public void TPMLP_MultiRanks_AllRanksProduceOutput()
        {
            var worldSize = 4;
            var outputs = new List<Tensor>();

            // Simulate execution on all ranks
            for (int rank = 0; rank < worldSize; rank++)
            {
                using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize, rank);

                var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
                var input = TPTestHelpers.CreateTestInput(batchSize: 4, inputSize: 10, seed: 42);

                var output = tpModel.Forward(input);
                outputs.Add(output);
            }

            // All ranks should produce output
            Assert.AreEqual(worldSize, outputs.Count);

            // All outputs should have the same shape
            for (int i = 1; i < outputs.Count; i++)
            {
                CollectionAssert.AreEqual(outputs[0].Shape, outputs[i].Shape);
            }
        }

        [TestMethod]
        public void TPMLP_WeightSharding_EachRankHasDifferentShard()
        {
            var worldSize = 4;
            var fc1Weights = new List<Tensor>();

            // Collect weights from all ranks
            for (int rank = 0; rank < worldSize; rank++)
            {
                using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize, rank);

                var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
                var fc1 = tpModel.GetModule("fc1") as MockColumnParallelLinear;

                Assert.IsNotNull(fc1);
                fc1Weights.Add(fc1.GetLocalWeight());
            }

            // Verify each rank has different weight values
            for (int i = 1; i < fc1Weights.Count; i++)
            {
                var diff = (fc1Weights[0] - fc1Weights[i]).Abs().Max().ToScalar();
                Assert.IsTrue(diff > 1e-6, "Different ranks should have different weight shards");
            }
        }

        [TestMethod]
        public void TPMLP_MultiRanks_ConsistentShapes()
        {
            var worldSize = 2;
            var shapes = new List<long[]>();

            for (int rank = 0; rank < worldSize; rank++)
            {
                using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize, rank);
                var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
                var input = TPTestHelpers.CreateTestInput(batchSize: 4, inputSize: 10);

                var output = tpModel.Forward(input);
                shapes.Add(output.Shape);
            }

            // All ranks should produce same output shape
            for (int i = 1; i < shapes.Count; i++)
            {
                CollectionAssert.AreEqual(shapes[0], shapes[i]);
            }
        }

        [TestMethod]
        public void TPMLP_TwoRanks_WeightPartitioning()
        {
            var worldSize = 2;
            var weightSizes = new List<long>();

            for (int rank = 0; rank < worldSize; rank++)
            {
                using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize, rank);
                var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
                var fc1 = tpModel.GetModule("fc1") as MockColumnParallelLinear;

                Assert.IsNotNull(fc1);
                var weight = fc1.GetLocalWeight();
                weightSizes.Add(weight.Shape[1]); // Output dimension is sharded
            }

            // Verify weight sizes sum to total
            var totalSize = weightSizes.Sum();
            Assert.AreEqual(20, totalSize, "Sharded weight sizes should sum to full size");

            // Each shard should have equal size
            Assert.AreEqual(weightSizes[0], weightSizes[1]);
        }

        [TestMethod]
        public void TPMLP_MultiRanks_BackwardPass()
        {
            var worldSize = 2;
            var gradients = new List<long[]>();

            for (int rank = 0; rank < worldSize; rank++)
            {
                using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize, rank);
                var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
                var input = TPTestHelpers.CreateTestInput(batchSize: 4, inputSize: 10);

                var output = tpModel.Forward(input);
                var gradOutput = Tensor.OnesLike(output);
                var gradInput = tpModel.Backward(gradOutput);

                gradients.Add(gradInput.Shape);
            }

            // All ranks should produce gradient inputs with same shape
            for (int i = 1; i < gradients.Count; i++)
            {
                CollectionAssert.AreEqual(gradients[0], gradients[i]);
            }
        }

        [TestMethod]
        public void TPMLP_FourRanks_DifferentInputs_SameModel()
        {
            var worldSize = 4;
            var outputs = new List<Tensor>();

            // Use same model context (would be same weights in real scenario)
            for (int rank = 0; rank < worldSize; rank++)
            {
                using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize, rank);
                var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
                var input = TPTestHelpers.CreateTestInput(batchSize: 4, inputSize: 10, seed: rank + 42);

                var output = tpModel.Forward(input);
                outputs.Add(output);
            }

            // Different inputs should produce different outputs
            var allDifferent = true;
            for (int i = 0; i < outputs.Count && allDifferent; i++)
            {
                for (int j = i + 1; j < outputs.Count && allDifferent; j++)
                {
                    allDifferent = !TPTestHelpers.TensorsApproxEqual(outputs[i], outputs[j], tolerance: 1e-10);
                }
            }
            Assert.IsTrue(allDifferent, "Different inputs should produce different outputs");
        }

        [TestMethod]
        public void TPMLP_RankIndex_BoundsCheck()
        {
            var worldSize = 4;

            // Test all valid ranks
            for (int rank = 0; rank < worldSize; rank++)
            {
                using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize, rank);
                Assert.AreEqual(rank, tpContext.Rank);
                Assert.AreEqual(worldSize, tpContext.WorldSize);

                var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
                var input = TPTestHelpers.CreateTestInput(batchSize: 4, inputSize: 10);
                var output = tpModel.Forward(input);

                Assert.IsNotNull(output);
            }
        }
    }
}
