using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Tests.Integration;
using MLFramework.Tensor;
using System;

namespace MLFramework.Tests.Integration
{
    [TestClass]
    public class TPTrainingTests
    {
        [TestMethod]
        public void TPMLP_Training_LossDecreases()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);

            var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
            var optimizer = new Adam(tpModel.Parameters, learningRate: 0.01);
            var lossFn = new MSELoss();

            var input = TPTestHelpers.CreateTestInput(batchSize: 32, inputSize: 10, seed: 42);
            var target = Tensor.Random(new long[] { 32, 5 }, random: new Random(123));

            // Train for a few iterations
            var initialLoss = TrainOneIteration(tpModel, optimizer, input, target, lossFn);

            for (int i = 0; i < 10; i++)
            {
                TrainOneIteration(tpModel, optimizer, input, target, lossFn);
            }

            var finalLoss = TrainOneIteration(tpModel, optimizer, input, target, lossFn);

            // Assert: Loss should decrease
            Assert.IsTrue(finalLoss.ToScalar() < initialLoss.ToScalar(),
                "Loss should decrease during training");
        }

        [TestMethod]
        public void TPMLP_Training_GradientsComputed()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);

            var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
            var optimizer = new Adam(tpModel.Parameters, learningRate: 0.01);
            var lossFn = new MSELoss();

            var input = TPTestHelpers.CreateTestInput(batchSize: 16, inputSize: 10);
            var target = Tensor.Random(new long[] { 16, 5 });

            // Act
            TrainOneIteration(tpModel, optimizer, input, target, lossFn);

            // Assert: Gradients should be computed for all parameters
            foreach (var param in tpModel.Parameters)
            {
                if (param.RequiresGrad)
                {
                    Assert.IsNotNull(param.Grad, $"Parameter {param.Name} should have gradients");
                    Assert.IsFalse(IsTensorAllZeros(param.Grad), $"Gradient for {param.Name} should not be all zeros");
                }
            }
        }

        [TestMethod]
        public void TPMLP_Training_MultipleSteps_ParametersChange()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);

            var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
            var optimizer = new Adam(tpModel.Parameters, learningRate: 0.01);
            var lossFn = new MSELoss();

            var input = TPTestHelpers.CreateTestInput(batchSize: 16, inputSize: 10);
            var target = Tensor.Random(new long[] { 16, 5 });

            // Get initial parameters
            var initialParams = new double[tpModel.Parameters.Count];
            for (int i = 0; i < tpModel.Parameters.Count; i++)
            {
                initialParams[i] = tpModel.Parameters[i].Data.Sum().ToScalar();
            }

            // Act: Train for multiple steps
            for (int i = 0; i < 5; i++)
            {
                TrainOneIteration(tpModel, optimizer, input, target, lossFn);
            }

            // Assert: Parameters should change
            bool paramsChanged = false;
            for (int i = 0; i < tpModel.Parameters.Count; i++)
            {
                var currentParamSum = tpModel.Parameters[i].Data.Sum().ToScalar();
                if (Math.Abs(currentParamSum - initialParams[i]) > 1e-6)
                {
                    paramsChanged = true;
                    break;
                }
            }
            Assert.IsTrue(paramsChanged, "Parameters should change after training");
        }

        [TestMethod]
        public void TPMLP_Training_SmallBatchSize_Works()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);

            var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
            var optimizer = new Adam(tpModel.Parameters, learningRate: 0.01);
            var lossFn = new MSELoss();

            var input = TPTestHelpers.CreateTestInput(batchSize: 2, inputSize: 10); // Very small batch
            var target = Tensor.Random(new long[] { 2, 5 });

            // Act
            var loss = TrainOneIteration(tpModel, optimizer, input, target, lossFn);

            // Assert
            Assert.IsNotNull(loss);
            Assert.IsTrue(loss.ToScalar() >= 0, "Loss should be non-negative");
        }

        [TestMethod]
        public void TPMLP_Training_ZeroGrad_ClearsGradients()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);

            var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
            var optimizer = new Adam(tpModel.Parameters, learningRate: 0.01);
            var lossFn = new MSELoss();

            var input = TPTestHelpers.CreateTestInput(batchSize: 16, inputSize: 10);
            var target = Tensor.Random(new long[] { 16, 5 });

            // First training step
            TrainOneIteration(tpModel, optimizer, input, target, lossFn);

            // Zero gradients
            optimizer.ZeroGrad();

            // Assert: All gradients should be zero
            foreach (var param in tpModel.Parameters)
            {
                if (param.RequiresGrad && param.Grad != null)
                {
                    Assert.IsTrue(IsTensorAllZeros(param.Grad), $"Gradients should be zero after ZeroGrad for {param.Name}");
                }
            }
        }

        [TestMethod]
        public void TPMLP_Training_EpochLoop_Completes()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);

            var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
            var optimizer = new Adam(tpModel.Parameters, learningRate: 0.01);
            var lossFn = new MSELoss();

            var input = TPTestHelpers.CreateTestInput(batchSize: 16, inputSize: 10);
            var target = Tensor.Random(new long[] { 16, 5 });

            int epochs = 3;
            int stepsPerEpoch = 5;
            double totalLoss = 0;

            // Act: Run multiple epochs
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int step = 0; step < stepsPerEpoch; step++)
                {
                    var loss = TrainOneIteration(tpModel, optimizer, input, target, lossFn);
                    totalLoss += loss.ToScalar();
                }
            }

            // Assert
            Assert.IsTrue(totalLoss > 0, "Training should complete and accumulate loss");
        }

        [TestMethod]
        public void TPMLP_Training_DifferentLearningRates_Work()
        {
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);

            double[] learningRates = { 0.001, 0.01, 0.1 };

            foreach (var lr in learningRates)
            {
                var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
                var optimizer = new Adam(tpModel.Parameters, learningRate: lr);
                var lossFn = new MSELoss();

                var input = TPTestHelpers.CreateTestInput(batchSize: 16, inputSize: 10);
                var target = Tensor.Random(new long[] { 16, 5 });

                // Act
                var loss = TrainOneIteration(tpModel, optimizer, input, target, lossFn);

                // Assert
                Assert.IsNotNull(loss);
                Assert.IsTrue(loss.ToScalar() >= 0, $"Training should work with learning rate {lr}");
            }
        }

        private double TrainOneIteration(
            Module model,
            Optimizer optimizer,
            Tensor input,
            Tensor target,
            Loss lossFn)
        {
            optimizer.ZeroGrad();

            var output = model.Forward(input);
            var loss = lossFn.Compute(output, target);

            var gradLoss = loss.Backward();
            model.Backward(gradLoss);

            optimizer.Step();

            return loss.ToScalar();
        }

        private bool IsTensorAllZeros(Tensor tensor)
        {
            var absSum = tensor.Abs().Sum().ToScalar();
            return absSum < 1e-10;
        }
    }
}
