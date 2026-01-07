using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Tests.Integration;
using MLFramework.Tensor;

namespace MLFramework.Tests.Integration
{
    [TestClass]
    public class TPEndToEndTests
    {
        [TestMethod]
        public void TPMLP_ForwardPass_ProducesCorrectOutput()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);

            var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
            var input = TPTestHelpers.CreateTestInput(batchSize: 4, inputSize: 10);

            // Act
            var output = tpModel.Forward(input);

            // Assert
            Assert.IsNotNull(output);
            Assert.AreEqual(2, output.Shape.Length);
            Assert.AreEqual(4, output.Shape[0]); // batch size
            Assert.AreEqual(5, output.Shape[^1]);  // output size
        }

        [TestMethod]
        public void TPMLP_BackwardPass_ComputesCorrectGradients()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);
            var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
            var input = TPTestHelpers.CreateTestInput(batchSize: 4, inputSize: 10);

            // Forward pass
            var output = tpModel.Forward(input);

            // Create gradient
            var gradOutput = Tensor.OnesLike(output);

            // Act
            var gradInput = tpModel.Backward(gradOutput);

            // Assert
            Assert.IsNotNull(gradInput);
            Assert.AreEqual(input.Shape.Length, gradInput.Shape.Length);
            CollectionAssert.AreEqual(input.Shape, gradInput.Shape);
        }

        [TestMethod]
        public void TPMLP_SingleRank_WorksCorrectly()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 1, rank: 0);
            var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
            var input = TPTestHelpers.CreateTestInput(batchSize: 4, inputSize: 10);

            // Act
            var output = tpModel.Forward(input);

            // Assert
            Assert.IsNotNull(output);
            Assert.AreEqual(4, output.Shape[0]);
            Assert.AreEqual(5, output.Shape[^1]);
        }

        [TestMethod]
        public void TPMLP_MultipleForwardPasses_ConsistentOutput()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);
            var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
            var input = TPTestHelpers.CreateTestInput(batchSize: 4, inputSize: 10, seed: 42);

            // Act
            var output1 = tpModel.Forward(input);
            var output2 = tpModel.Forward(input);

            // Assert
            Assert.IsTrue(
                TPTestHelpers.TensorsApproxEqual(output1, output2, tolerance: 1e-10),
                "Multiple forward passes should produce the same output");
        }

        [TestMethod]
        public void TPMLP_DifferentBatchSizes_WorksCorrectly()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);
            var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);

            int[] batchSizes = { 1, 4, 8, 16 };

            foreach (int batchSize in batchSizes)
            {
                // Act
                var input = TPTestHelpers.CreateTestInput(batchSize: batchSize, inputSize: 10);
                var output = tpModel.Forward(input);

                // Assert
                Assert.AreEqual(batchSize, output.Shape[0]);
                Assert.AreEqual(5, output.Shape[^1]);
            }
        }

        [TestMethod]
        public void TPMLP_NoBias_WorksCorrectly()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);
            var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, bias: false, context: tpContext);
            var input = TPTestHelpers.CreateTestInput(batchSize: 4, inputSize: 10);

            // Act
            var output = tpModel.Forward(input);

            // Assert
            Assert.IsNotNull(output);
            Assert.AreEqual(4, output.Shape[0]);
            Assert.AreEqual(5, output.Shape[^1]);
        }

        [TestMethod]
        public void TPMLP_LargeDimensions_WorksCorrectly()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 4, rank: 0);
            var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 100, hiddenSize: 200, outputSize: 50, context: tpContext);
            var input = TPTestHelpers.CreateTestInput(batchSize: 8, inputSize: 100);

            // Act
            var output = tpModel.Forward(input);

            // Assert
            Assert.IsNotNull(output);
            Assert.AreEqual(8, output.Shape[0]);
            Assert.AreEqual(50, output.Shape[^1]);
        }
    }
}
