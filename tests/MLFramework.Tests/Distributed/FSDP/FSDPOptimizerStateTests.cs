using Microsoft.VisualStudio.TestTools.UnitTesting;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using MLFramework.Distributed.FSDP;
using System;

namespace MLFramework.Tests.Distributed.FSDP
{
    /// <summary>
    /// Unit tests for FSDP optimizer state management.
    /// </summary>
    [TestClass]
    public class FSDPOptimizerStateTests
    {
        [TestMethod]
        public void TestAdamOptimizerStateCreation()
        {
            var param = Tensor.Zeros(new[] { 100L }, DataType.Float32);
            var state = new AdamOptimizerState(param, 0, 4);

            Assert.AreEqual(OptimizerStateType.Adam, state.StateType);
            Assert.AreEqual(0, state.ShardIndex);
            Assert.AreEqual(4, state.NumShards);
            Assert.AreEqual(0, state.StepCount);
            Assert.IsNotNull(state.MomentumBuffer);
            Assert.IsNotNull(state.VarianceBuffer);
            Assert.AreEqual(100, state.MomentumBuffer.Size);
            Assert.AreEqual(100, state.VarianceBuffer.Size);
        }

        [TestMethod]
        public void TestSGDOptimizerStateCreation()
        {
            var state = new SGDOptimizerState(0, 4);

            Assert.AreEqual(OptimizerStateType.SGD, state.StateType);
            Assert.AreEqual(0, state.ShardIndex);
            Assert.AreEqual(4, state.NumShards);
            Assert.AreEqual(0, state.StepCount);
        }

        [TestMethod]
        public void TestAdamWOptimizerStateCreation()
        {
            var param = Tensor.Zeros(new[] { 100L }, DataType.Float32);
            var state = new AdamWOptimizerState(param, 0, 4);

            Assert.AreEqual(OptimizerStateType.AdamW, state.StateType);
            Assert.AreEqual(0, state.ShardIndex);
            Assert.AreEqual(4, state.NumShards);
            Assert.AreEqual(0, state.StepCount);
            Assert.IsNotNull(state.MomentumBuffer);
            Assert.IsNotNull(state.VarianceBuffer);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TestAdamOptimizerStateNullParameter()
        {
            var state = new AdamOptimizerState(null, 0, 4);
        }

        [TestMethod]
        public void TestAdamOptimizerStateCloning()
        {
            var param = Tensor.Zeros(new[] { 100L }, DataType.Float32);
            var state = new AdamOptimizerState(param, 0, 4);
            state.StepCount = 10;

            var cloned = state.Clone() as AdamOptimizerState;

            Assert.IsNotNull(cloned);
            Assert.AreEqual(state.ShardIndex, cloned.ShardIndex);
            Assert.AreEqual(state.NumShards, cloned.NumShards);
            Assert.AreEqual(state.StepCount, cloned.StepCount);
            Assert.AreEqual(state.MomentumBuffer.Size, cloned.MomentumBuffer.Size);
            Assert.AreEqual(state.VarianceBuffer.Size, cloned.VarianceBuffer.Size);
            Assert.AreEqual(state.StateType, cloned.StateType);
        }

        [TestMethod]
        public void TestSGDOptimizerStateCloning()
        {
            var state = new SGDOptimizerState(2, 8);
            state.StepCount = 5;

            var cloned = state.Clone() as SGDOptimizerState;

            Assert.IsNotNull(cloned);
            Assert.AreEqual(state.ShardIndex, cloned.ShardIndex);
            Assert.AreEqual(state.NumShards, cloned.NumShards);
            Assert.AreEqual(state.StepCount, cloned.StepCount);
            Assert.AreEqual(state.StateType, cloned.StateType);
        }

        [TestMethod]
        public void TestAdamWOptimizerStateCloning()
        {
            var param = Tensor.Zeros(new[] { 100L }, DataType.Float32);
            var state = new AdamWOptimizerState(param, 1, 4);
            state.StepCount = 15;

            var cloned = state.Clone() as AdamWOptimizerState;

            Assert.IsNotNull(cloned);
            Assert.AreEqual(OptimizerStateType.AdamW, cloned.StateType);
            Assert.AreEqual(state.ShardIndex, cloned.ShardIndex);
            Assert.AreEqual(state.NumShards, cloned.NumShards);
            Assert.AreEqual(state.StepCount, cloned.StepCount);
        }

        [TestMethod]
        public void TestOptimizerStateDisposal()
        {
            var param = Tensor.Zeros(new[] { 100L }, DataType.Float32);
            var state = new AdamOptimizerState(param, 0, 4);

            state.Dispose();

            Assert.IsNull(state.MomentumBuffer);
            Assert.IsNull(state.VarianceBuffer);
        }

        [TestMethod]
        public void TestSGDOptimizerStateDisposal()
        {
            var state = new SGDOptimizerState(0, 4);

            // Should not throw
            state.Dispose();
            state.Dispose();
        }

        [TestMethod]
        public void TestAdamOptimizerStateWithExplicitSize()
        {
            var state = new AdamOptimizerState(50, DataType.Float32, 1, 4);

            Assert.AreEqual(OptimizerStateType.Adam, state.StateType);
            Assert.AreEqual(1, state.ShardIndex);
            Assert.AreEqual(4, state.NumShards);
            Assert.IsNotNull(state.MomentumBuffer);
            Assert.IsNotNull(state.VarianceBuffer);
            Assert.AreEqual(50, state.MomentumBuffer.Size);
            Assert.AreEqual(50, state.VarianceBuffer.Size);
        }

        [TestMethod]
        public void TestAdamWOptimizerStateWithExplicitSize()
        {
            var state = new AdamWOptimizerState(75, DataType.Float32, 2, 8);

            Assert.AreEqual(OptimizerStateType.AdamW, state.StateType);
            Assert.AreEqual(2, state.ShardIndex);
            Assert.AreEqual(8, state.NumShards);
            Assert.IsNotNull(state.MomentumBuffer);
            Assert.IsNotNull(state.VarianceBuffer);
            Assert.AreEqual(75, state.MomentumBuffer.Size);
            Assert.AreEqual(75, state.VarianceBuffer.Size);
        }

        [TestMethod]
        public void TestStepCountIncrement()
        {
            var param = Tensor.Zeros(new[] { 100L }, DataType.Float32);
            var state = new AdamOptimizerState(param, 0, 4);

            Assert.AreEqual(0, state.StepCount);

            state.StepCount = 1;
            Assert.AreEqual(1, state.StepCount);

            state.StepCount = 100;
            Assert.AreEqual(100, state.StepCount);
        }

        [TestMethod]
        public void TestMultipleOptimizerStates()
        {
            var param1 = Tensor.Zeros(new[] { 100L }, DataType.Float32);
            var param2 = Tensor.Zeros(new[] { 200L }, DataType.Float32);

            var state1 = new AdamOptimizerState(param1, 0, 4);
            var state2 = new AdamOptimizerState(param2, 1, 4);

            Assert.AreEqual(0, state1.ShardIndex);
            Assert.AreEqual(1, state2.ShardIndex);
            Assert.AreEqual(100, state1.MomentumBuffer.Size);
            Assert.AreEqual(200, state2.MomentumBuffer.Size);
        }

        [TestMethod]
        public void TestDifferentDataTypes()
        {
            // Float32
            var floatParam = Tensor.Zeros(new[] { 100L }, DataType.Float32);
            var floatState = new AdamOptimizerState(floatParam, 0, 4);
            Assert.AreEqual(DataType.Float32, floatState.MomentumBuffer.Dtype);

            // Float16
            var halfParam = Tensor.Zeros(new[] { 100L }, DataType.Float16);
            var halfState = new AdamOptimizerState(halfParam, 0, 4);
            Assert.AreEqual(DataType.Float16, halfState.MomentumBuffer.Dtype);
        }

        [TestMethod]
        public void TestClonedStateIsIndependent()
        {
            var param = Tensor.Zeros(new[] { 100L }, DataType.Float32);
            var state = new AdamOptimizerState(param, 0, 4);
            state.StepCount = 10;

            var cloned = state.Clone() as AdamOptimizerState;

            // Modify original
            state.StepCount = 20;

            // Cloned should be independent
            Assert.AreEqual(10, cloned.StepCount);
        }

        [TestMethod]
        public void TestOptimizerStateAfterDispose()
        {
            var param = Tensor.Zeros(new[] { 100L }, DataType.Float32);
            var state = new AdamOptimizerState(param, 0, 4);

            state.Dispose();

            // These should be null after dispose
            Assert.IsNull(state.MomentumBuffer);
            Assert.IsNull(state.VarianceBuffer);

            // But other properties should still be accessible
            Assert.AreEqual(OptimizerStateType.Adam, state.StateType);
            Assert.AreEqual(0, state.ShardIndex);
            Assert.AreEqual(4, state.NumShards);
        }
    }
}
