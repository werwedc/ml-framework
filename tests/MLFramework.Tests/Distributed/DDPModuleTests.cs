using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Tensor;
using System;

namespace MLFramework.Tests.Distributed
{
    /// <summary>
    /// Mock DistributedDataParallel for testing.
    /// </summary>
    public class DistributedDataParallel
    {
        private readonly SimpleModel _model;
        private readonly IProcessGroup _processGroup;

        public DistributedDataParallel(SimpleModel model, IProcessGroup processGroup)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _processGroup = processGroup ?? throw new ArgumentNullException(nameof(processGroup));
        }

        public Tensor Forward(Tensor input)
        {
            return _model.Forward(input);
        }

        public void BroadcastParameters()
        {
            // In a real implementation, this would broadcast from rank 0
            // For mock testing, we simulate it
            if (_processGroup.Rank == 0)
            {
                // Rank 0 broadcasts its parameters
                _processGroup.Broadcast(_model.Weight, 0);
                _processGroup.Broadcast(_model.Bias, 0);
            }
            else
            {
                // Other ranks receive the broadcast
                _processGroup.Broadcast(_model.Weight, 0);
                _processGroup.Broadcast(_model.Bias, 0);
            }
        }

        public SimpleModel Model => _model;
    }

    [TestClass]
    public class DDPModuleTests
    {
        [TestMethod]
        public void DDP_Forward_PreservesModuleBehavior()
        {
            var processGroup = MockProcessGroup.Create(worldSize: 2, rank: 0);
            var model = new SimpleModel();
            var ddpModel = new DistributedDataParallel(model, processGroup);

            var input = Tensor.Random(new long[] { 10, 10 });
            var output = ddpModel.Forward(input);

            Assert.IsNotNull(output);

            processGroup.Destroy();
        }

        [TestMethod]
        public void DDP_BroadcastParameters_SynchronizesWeights()
        {
            var processGroup1 = MockProcessGroup.Create(worldSize: 2, rank: 0);
            var processGroup2 = MockProcessGroup.Create(worldSize: 2, rank: 1);

            var model1 = new SimpleModel();
            var model2 = new SimpleModel();

            // Make models have different weights
            model1.Weight.Fill_(1.0);
            model2.Weight.Fill_(2.0);

            var ddp1 = new DistributedDataParallel(model1, processGroup1);
            var ddp2 = new DistributedDataParallel(model2, processGroup2);

            // Broadcast from rank 0
            ddp1.BroadcastParameters();
            ddp2.BroadcastParameters();

            // Both models should now have the same weights (from rank 0)
            Assert.IsTrue(Tensor.AllClose(model1.Weight, model2.Weight), "Weights should be synchronized after broadcast");

            processGroup1.Destroy();
            processGroup2.Destroy();
        }

        [TestMethod]
        public void DDP_Constructor_NullModel_ThrowsException()
        {
            var processGroup = MockProcessGroup.Create(worldSize: 2, rank: 0);

            Assert.ThrowsException<ArgumentNullException>(() =>
            {
                var ddp = new DistributedDataParallel(null, processGroup);
            });

            processGroup.Destroy();
        }

        [TestMethod]
        public void DDP_Constructor_NullProcessGroup_ThrowsException()
        {
            var model = new SimpleModel();

            Assert.ThrowsException<ArgumentNullException>(() =>
            {
                var ddp = new DistributedDataParallel(model, null);
            });
        }

        [TestMethod]
        public void DDP_Forward_ProducesCorrectOutputShape()
        {
            var processGroup = MockProcessGroup.Create(worldSize: 2, rank: 0);
            var model = new SimpleModel();
            var ddpModel = new DistributedDataParallel(model, processGroup);

            var input = Tensor.Random(new long[] { 5, 10 });
            var output = ddpModel.Forward(input);

            // Output shape should be (5, 10) based on SimpleModel
            Assert.AreEqual(2, output.Shape.Length);
            Assert.AreEqual(5, output.Shape[0]);
            Assert.AreEqual(10, output.Shape[1]);

            processGroup.Destroy();
        }

        [TestMethod]
        public void DDP_ModelProperty_ReturnsOriginalModel()
        {
            var processGroup = MockProcessGroup.Create(worldSize: 2, rank: 0);
            var model = new SimpleModel();
            var ddpModel = new DistributedDataParallel(model, processGroup);

            Assert.AreSame(model, ddpModel.Model);

            processGroup.Destroy();
        }

        [TestMethod]
        public void DDP_MultipleForwards_ProduceDifferentOutputs()
        {
            var processGroup = MockProcessGroup.Create(worldSize: 2, rank: 0);
            var model = new SimpleModel();
            var ddpModel = new DistributedDataParallel(model, processGroup);

            var input1 = Tensor.Random(new long[] { 10, 10 });
            var input2 = Tensor.Random(new long[] { 10, 10 });

            var output1 = ddpModel.Forward(input1);
            var output2 = ddpModel.Forward(input2);

            // With different inputs, outputs should be different
            Assert.IsFalse(Tensor.AllClose(output1, output2), "Different inputs should produce different outputs");

            processGroup.Destroy();
        }

        [TestMethod]
        public void DDP_BroadcastParameters_WithDifferentBiases_Synchronizes()
        {
            var processGroup1 = MockProcessGroup.Create(worldSize: 2, rank: 0);
            var processGroup2 = MockProcessGroup.Create(worldSize: 2, rank: 1);

            var model1 = new SimpleModel();
            var model2 = new SimpleModel();

            // Make models have different biases
            model1.Bias.Fill_(1.0);
            model2.Bias.Fill_(3.0);

            var ddp1 = new DistributedDataParallel(model1, processGroup1);
            var ddp2 = new DistributedDataParallel(model2, processGroup2);

            ddp1.BroadcastParameters();
            ddp2.BroadcastParameters();

            // Both models should now have the same biases
            Assert.IsTrue(Tensor.AllClose(model1.Bias, model2.Bias), "Biases should be synchronized after broadcast");

            processGroup1.Destroy();
            processGroup2.Destroy();
        }
    }
}
