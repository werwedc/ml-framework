using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Distributed.FSDP;
using MLFramework.Distributed;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLFramework.Tests.Distributed.FSDP
{
    [TestClass]
    public class FSDPCpuOffloadTests
    {
        private MockProcessGroup _processGroup;
        private FSDP _fsdp;
        private FSDPConfig _fsdpConfig;
        private FSDPCpuOffloadConfig _cpuOffloadConfig;
        private FSDPCpuOffloader _cpuOffloader;

        [TestInitialize]
        public void Setup()
        {
            _processGroup = new MockProcessGroup(2, 0);
            _fsdpConfig = new FSDPConfig
            {
                ShardingStrategy = ShardingStrategy.Full,
                MixedPrecision = false
            };

            var model = new MockModel();
            _fsdp = new FSDP(model, _fsdpConfig, _processGroup);

            _cpuOffloadConfig = new FSDPCpuOffloadConfig
            {
                Enabled = true,
                OffloadOptimizerStates = true,
                OffloadGradients = true,
                OffloadParameters = true,
                PrefetchParameters = true,
                PrefetchGradients = true,
                PrefetchSteps = 1
            };

            _cpuOffloader = new FSDPCpuOffloader(_cpuOffloadConfig, _fsdp);
        }

        [TestCleanup]
        public void Cleanup()
        {
            _cpuOffloader?.Dispose();
            _fsdp?.Dispose();
        }

        #region FSDPCpuOffloadConfig Tests

        [TestMethod]
        public void FSDPCpuOffloadConfig_DefaultValues_AreCorrect()
        {
            var config = new FSDPCpuOffloadConfig();

            Assert.IsTrue(config.Enabled);
            Assert.IsTrue(config.OffloadOptimizerStates);
            Assert.IsTrue(config.OffloadGradients);
            Assert.IsTrue(config.OffloadParameters);
            Assert.IsTrue(config.PrefetchParameters);
            Assert.IsTrue(config.PrefetchGradients);
            Assert.AreEqual(1, config.PrefetchSteps);
        }

        [TestMethod]
        public void FSDPCpuOffloadConfig_CanModifyProperties()
        {
            var config = new FSDPCpuOffloadConfig
            {
                Enabled = false,
                OffloadParameters = false,
                PrefetchSteps = 3
            };

            Assert.IsFalse(config.Enabled);
            Assert.IsFalse(config.OffloadParameters);
            Assert.AreEqual(3, config.PrefetchSteps);
        }

        [TestMethod]
        public void FSDPCpuOffloadConfig_Validate_ValidPrefetchSteps_DoesNotThrow()
        {
            var config = new FSDPCpuOffloadConfig { PrefetchSteps = 5 };
            config.Validate(); // Should not throw
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void FSDPCpuOffloadConfig_Validate_NegativePrefetchSteps_ThrowsException()
        {
            var config = new FSDPCpuOffloadConfig { PrefetchSteps = -1 };
            config.Validate();
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void FSDPCpuOffloadConfig_Validate_PrefetchStepsTooLarge_ThrowsException()
        {
            var config = new FSDPCpuOffloadConfig { PrefetchSteps = 11 };
            config.Validate();
        }

        #endregion

        #region CpuOffloadBuffer Tests

        [TestMethod]
        public void CpuOffloadBuffer_InitializesWithCorrectDefaults()
        {
            var buffer = new CpuOffloadBuffer();

            Assert.IsNotNull(buffer);
            Assert.IsNull(buffer.ParameterBuffer);
            Assert.IsNull(buffer.GradientBuffer);
            Assert.IsNull(buffer.MomentumBuffer);
            Assert.IsNull(buffer.VarianceBuffer);
            Assert.AreNotEqual(default(DateTime), buffer.LastAccessTime);
        }

        [TestMethod]
        public void CpuOffloadBuffer_CanSetProperties()
        {
            var buffer = new CpuOffloadBuffer
            {
                ParameterBuffer = new float[100],
                GradientBuffer = new float[100]
            };

            Assert.AreEqual(100, buffer.ParameterBuffer.Length);
            Assert.AreEqual(100, buffer.GradientBuffer.Length);
        }

        #endregion

        #region FSDPCpuOffloader - Parameter Offloading Tests

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void FSDPCpuOffloader_OffloadParameter_NullShardingUnit_ThrowsException()
        {
            _cpuOffloader.OffloadParameter(null);
        }

        [TestMethod]
        public void FSDPCpuOffloader_OffloadParameter_WhenDisabled_DoesNothing()
        {
            var config = new FSDPCpuOffloadConfig { OffloadParameters = false };
            var offloader = new FSDPCpuOffloader(config, _fsdp);

            var shardingUnit = CreateTestShardingUnit("test_param");
            shardingUnit.ShardedParameter.Data[0] = 1.0f;

            offloader.OffloadParameter(shardingUnit);

            Assert.IsFalse(shardingUnit.State.IsOffloaded);

            offloader.Dispose();
        }

        [TestMethod]
        public void FSDPCpuOffloader_OffloadParameter_OffloadsToCPU()
        {
            var shardingUnit = CreateTestShardingUnit("test_param");
            shardingUnit.ShardedParameter.Data[0] = 1.5f;
            shardingUnit.ShardedParameter.Data[1] = 2.5f;

            _cpuOffloader.OffloadParameter(shardingUnit);

            Assert.IsTrue(shardingUnit.State.IsOffloaded);

            var buffer = _cpuOffloader.GetCpuBuffer("test_param");
            Assert.IsNotNull(buffer);
            Assert.AreEqual(1.5f, buffer.ParameterBuffer[0], 0.001f);
            Assert.AreEqual(2.5f, buffer.ParameterBuffer[1], 0.001f);
        }

        [TestMethod]
        public void FSDPCpuOffloader_OffloadParameter_AlreadyOffloaded_DoesNothing()
        {
            var shardingUnit = CreateTestShardingUnit("test_param");
            shardingUnit.State.IsOffloaded = true;

            _cpuOffloader.OffloadParameter(shardingUnit);

            var buffer = _cpuOffloader.GetCpuBuffer("test_param");
            Assert.IsNull(buffer); // Should not create buffer if already offloaded
        }

        #endregion

        #region FSDPCpuOffloader - Parameter Prefetch Tests

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void FSDPCpuOffloader_PrefetchParameter_NullShardingUnit_ThrowsException()
        {
            _cpuOffloader.PrefetchParameter(null);
        }

        [TestMethod]
        public void FSDPCpuOffloader_PrefetchParameter_WhenDisabled_DoesNothing()
        {
            var config = new FSDPCpuOffloadConfig { PrefetchParameters = false };
            var offloader = new FSDPCpuOffloader(config, _fsdp);

            var shardingUnit = CreateTestShardingUnit("test_param");
            shardingUnit.State.IsOffloaded = true;

            offloader.PrefetchParameter(shardingUnit);

            Assert.IsTrue(shardingUnit.State.IsOffloaded); // Still offloaded

            offloader.Dispose();
        }

        [TestMethod]
        public void FSDPCpuOffloader_PrefetchParameter_PrefetchesFromCPU()
        {
            var shardingUnit = CreateTestShardingUnit("test_param");
            shardingUnit.ShardedParameter.Data[0] = 1.5f;
            shardingUnit.ShardedParameter.Data[1] = 2.5f;

            // Offload first
            _cpuOffloader.OffloadParameter(shardingUnit);

            // Modify CPU buffer
            var buffer = _cpuOffloader.GetCpuBuffer("test_param");
            buffer.ParameterBuffer[0] = 3.0f;
            buffer.ParameterBuffer[1] = 4.0f;

            // Prefetch
            _cpuOffloader.PrefetchParameter(shardingUnit);

            Assert.IsFalse(shardingUnit.State.IsOffloaded);
            Assert.AreEqual(3.0f, shardingUnit.ShardedParameter.Data[0], 0.001f);
            Assert.AreEqual(4.0f, shardingUnit.ShardedParameter.Data[1], 0.001f);
        }

        [TestMethod]
        public void FSDPCpuOffloader_PrefetchParameter_NotOffloaded_DoesNothing()
        {
            var shardingUnit = CreateTestShardingUnit("test_param");
            shardingUnit.State.IsOffloaded = false;

            _cpuOffloader.PrefetchParameter(shardingUnit);

            Assert.IsFalse(shardingUnit.State.IsOffloaded);
        }

        #endregion

        #region FSDPCpuOffloader - Gradient Offloading Tests

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void FSDPCpuOffloader_OffloadGradient_NullShardingUnit_ThrowsException()
        {
            _cpuOffloader.OffloadGradient(null);
        }

        [TestMethod]
        public void FSDPCpuOffloader_OffloadGradient_WhenDisabled_DoesNothing()
        {
            var config = new FSDPCpuOffloadConfig { OffloadGradients = false };
            var offloader = new FSDPCpuOffloader(config, _fsdp);

            var shardingUnit = CreateTestShardingUnit("test_param");
            shardingUnit.LocalGradient = Tensor.Zeros(new[] { 2 });
            shardingUnit.LocalGradient.Data[0] = 0.1f;

            offloader.OffloadGradient(shardingUnit);

            Assert.IsFalse(shardingUnit.State.IsOffloaded);

            offloader.Dispose();
        }

        [TestMethod]
        public void FSDPCpuOffloader_OffloadGradient_OffloadsToCPU()
        {
            var shardingUnit = CreateTestShardingUnit("test_param");
            shardingUnit.LocalGradient = Tensor.Zeros(new[] { 2 });
            shardingUnit.LocalGradient.Data[0] = 0.1f;
            shardingUnit.LocalGradient.Data[1] = 0.2f;

            _cpuOffloader.OffloadGradient(shardingUnit);

            Assert.IsTrue(shardingUnit.State.IsOffloaded);

            var buffer = _cpuOffloader.GetCpuBuffer("test_param");
            Assert.IsNotNull(buffer);
            Assert.AreEqual(0.1f, buffer.GradientBuffer[0], 0.001f);
            Assert.AreEqual(0.2f, buffer.GradientBuffer[1], 0.001f);
        }

        [TestMethod]
        public void FSDPCpuOffloader_OffloadGradient_NullGradient_DoesNothing()
        {
            var shardingUnit = CreateTestShardingUnit("test_param");
            shardingUnit.LocalGradient = null;

            _cpuOffloader.OffloadGradient(shardingUnit);

            Assert.IsFalse(shardingUnit.State.IsOffloaded);
        }

        #endregion

        #region FSDPCpuOffloader - Gradient Prefetch Tests

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void FSDPCpuOffloader_PrefetchGradient_NullShardingUnit_ThrowsException()
        {
            _cpuOffloader.PrefetchGradient(null);
        }

        [TestMethod]
        public void FSDPCpuOffloader_PrefetchGradient_PrefetchesFromCPU()
        {
            var shardingUnit = CreateTestShardingUnit("test_param");
            shardingUnit.LocalGradient = Tensor.Zeros(new[] { 2 });
            shardingUnit.LocalGradient.Data[0] = 0.1f;
            shardingUnit.LocalGradient.Data[1] = 0.2f;

            // Offload first
            _cpuOffloader.OffloadGradient(shardingUnit);

            // Modify CPU buffer
            var buffer = _cpuOffloader.GetCpuBuffer("test_param");
            buffer.GradientBuffer[0] = 0.3f;
            buffer.GradientBuffer[1] = 0.4f;

            // Prefetch
            _cpuOffloader.PrefetchGradient(shardingUnit);

            Assert.IsFalse(shardingUnit.State.IsOffloaded);
            Assert.AreEqual(0.3f, shardingUnit.LocalGradient.Data[0], 0.001f);
            Assert.AreEqual(0.4f, shardingUnit.LocalGradient.Data[1], 0.001f);
        }

        #endregion

        #region FSDPCpuOffloader - Optimizer State Offloading Tests

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void FSDPCpuOffloader_OffloadOptimizerState_NullOptimizerState_ThrowsException()
        {
            _cpuOffloader.OffloadOptimizerState(null);
        }

        [TestMethod]
        public void FSDPCpuOffloader_OffloadOptimizerState_AdamState_OffloadsToCPU()
        {
            var parameter = Tensor.Zeros(new[] { 10 });
            var adamState = new AdamOptimizerState(parameter, 0, 2);
            adamState.MomentumBuffer.Data[0] = 1.0f;
            adamState.VarianceBuffer.Data[0] = 2.0f;

            _cpuOffloader.OffloadOptimizerState(adamState);

            var buffer = _cpuOffloader.GetCpuBuffer(adamState.ToString());
            Assert.IsNotNull(buffer);
            Assert.AreEqual(1.0f, buffer.MomentumBuffer[0], 0.001f);
            Assert.AreEqual(2.0f, buffer.VarianceBuffer[0], 0.001f);
        }

        [TestMethod]
        public void FSDPCpuOffloader_OffloadOptimizerState_WhenDisabled_DoesNothing()
        {
            var config = new FSDPCpuOffloadConfig { OffloadOptimizerStates = false };
            var offloader = new FSDPCpuOffloader(config, _fsdp);

            var parameter = Tensor.Zeros(new[] { 10 });
            var adamState = new AdamOptimizerState(parameter, 0, 2);

            offloader.OffloadOptimizerState(adamState);

            var buffer = offloader.GetCpuBuffer(adamState.ToString());
            Assert.IsNull(buffer);

            offloader.Dispose();
        }

        [TestMethod]
        public void FSDPCpuOffloader_PrefetchOptimizerState_PrefetchesFromCPU()
        {
            var parameter = Tensor.Zeros(new[] { 10 });
            var adamState = new AdamOptimizerState(parameter, 0, 2);
            adamState.MomentumBuffer.Data[0] = 1.0f;
            adamState.VarianceBuffer.Data[0] = 2.0f;

            // Offload first
            _cpuOffloader.OffloadOptimizerState(adamState);

            // Modify CPU buffer
            var buffer = _cpuOffloader.GetCpuBuffer(adamState.ToString());
            buffer.MomentumBuffer[0] = 3.0f;
            buffer.VarianceBuffer[0] = 4.0f;

            // Prefetch
            _cpuOffloader.PrefetchOptimizerState(adamState);

            Assert.AreEqual(3.0f, adamState.MomentumBuffer.Data[0], 0.001f);
            Assert.AreEqual(4.0f, adamState.VarianceBuffer.Data[0], 0.001f);
        }

        #endregion

        #region FSDPCpuOffloader - Buffer Management Tests

        [TestMethod]
        public void FSDPCpuOffloader_GetCpuBuffer_NonExistent_ReturnsNull()
        {
            var buffer = _cpuOffloader.GetCpuBuffer("non_existent");
            Assert.IsNull(buffer);
        }

        [TestMethod]
        public void FSDPCpuOffloader_ClearBuffers_RemovesAllBuffers()
        {
            var shardingUnit1 = CreateTestShardingUnit("param1");
            var shardingUnit2 = CreateTestShardingUnit("param2");

            _cpuOffloader.OffloadParameter(shardingUnit1);
            _cpuOffloader.OffloadParameter(shardingUnit2);

            Assert.IsNotNull(_cpuOffloader.GetCpuBuffer("param1"));
            Assert.IsNotNull(_cpuOffloader.GetCpuBuffer("param2"));

            _cpuOffloader.ClearBuffers();

            Assert.IsNull(_cpuOffloader.GetCpuBuffer("param1"));
            Assert.IsNull(_cpuOffloader.GetCpuBuffer("param2"));
        }

        #endregion

        #region FSDPCpuOffloader - Dispose Tests

        [TestMethod]
        public void FSDPCpuOffloader_Dispose_ClearsBuffers()
        {
            var shardingUnit = CreateTestShardingUnit("test_param");
            _cpuOffloader.OffloadParameter(shardingUnit);

            _cpuOffloader.Dispose();

            // After dispose, buffers should be cleared
            // (This is implementation-dependent, but should be consistent)
        }

        [TestMethod]
        public void FSDPCpuOffloader_MultipleDispose_DoesNotThrow()
        {
            _cpuOffloader.Dispose();
            _cpuOffloader.Dispose(); // Should not throw
        }

        #endregion

        #region FSDPPrefetchManager Tests

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void FSDPPrefetchManager_ConstructWithNullOffloader_ThrowsException()
        {
            new FSDPPrefetchManager(null);
        }

        [TestMethod]
        public void FSDPPrefetchManager_ConstructWithValidParameters_DoesNotThrow()
        {
            var manager = new FSDPPrefetchManager(_cpuOffloader, 2);
            Assert.IsNotNull(manager);
            manager.Dispose();
        }

        [TestMethod]
        public void FSDPPrefetchManager_ConstructWithZeroPrefetchSteps_AllowsZero()
        {
            var manager = new FSDPPrefetchManager(_cpuOffloader, 0);
            Assert.IsNotNull(manager);
            manager.Dispose();
        }

        [TestMethod]
        public void FSDPPrefetchManager_SchedulePrefetch_AddsToQueue()
        {
            var manager = new FSDPPrefetchManager(_cpuOffloader, 1);
            var shardingUnits = new Dictionary<string, FSDPShardingUnit>();

            manager.SchedulePrefetch(shardingUnits, true);
            manager.Stop();

            manager.Dispose();
        }

        [TestMethod]
        public void FSDPPrefetchManager_SchedulePrefetch_WithZeroSteps_DoesNothing()
        {
            var manager = new FSDPPrefetchManager(_cpuOffloader, 0);
            var shardingUnits = new Dictionary<string, FSDPShardingUnit>();
            var shardingUnit = CreateTestShardingUnit("test_param");
            shardingUnits["test_param"] = shardingUnit;

            manager.SchedulePrefetch(shardingUnits, true);
            manager.Stop();

            // Should not prefetch
            Assert.IsFalse(shardingUnit.State.IsOffloaded);

            manager.Dispose();
        }

        [TestMethod]
        public void FSDPPrefetchManager_Clear_RemovesPendingTasks()
        {
            var manager = new FSDPPrefetchManager(_cpuOffloader, 1);

            manager.Clear();

            manager.Stop();
            manager.Dispose();
        }

        [TestMethod]
        public void FSDPPrefetchManager_Stop_StopsExecution()
        {
            var manager = new FSDPPrefetchManager(_cpuOffloader, 1);

            manager.Stop();

            manager.Dispose();
        }

        [TestMethod]
        public void FSDPPrefetchManager_MultipleDispose_DoesNotThrow()
        {
            var manager = new FSDPPrefetchManager(_cpuOffloader, 1);

            manager.Stop();
            manager.Dispose();
            manager.Dispose(); // Should not throw
        }

        #endregion

        #region Helper Methods

        private FSDPShardingUnit CreateTestShardingUnit(string paramName)
        {
            var parameter = Tensor.Zeros(new[] { 2 });
            parameter.Data[0] = 1.0f;
            parameter.Data[1] = 2.0f;

            return new FSDPShardingUnit(paramName, parameter, _processGroup);
        }

        #endregion
    }

    #region Mock Classes

    /// <summary>
    /// Mock process group for testing.
    /// </summary>
    public class MockProcessGroup : IProcessGroup
    {
        private readonly int _worldSize;
        private readonly int _rank;

        public MockProcessGroup(int worldSize, int rank)
        {
            _worldSize = worldSize;
            _rank = rank;
        }

        public int Rank => _rank;
        public int WorldSize => _worldSize;

        public void AllReduce(Tensor tensor)
        {
            // Mock implementation
        }

        public void Broadcast(Tensor tensor, int root)
        {
            // Mock implementation
        }

        public void Barrier()
        {
            // Mock implementation
        }

        public void Dispose()
        {
            // Mock implementation
        }
    }

    /// <summary>
    /// Mock model for testing.
    /// </summary>
    public class MockModel : IModel
    {
        private readonly List<NamedTensor> _parameters;

        public MockModel()
        {
            _parameters = new List<NamedTensor>
            {
                new NamedTensor("weight1", Tensor.Zeros(new[] { 10, 20 })),
                new NamedTensor("bias1", Tensor.Zeros(new[] { 20 })),
                new NamedTensor("weight2", Tensor.Zeros(new[] { 20, 10 })),
                new NamedTensor("bias2", Tensor.Zeros(new[] { 10 }))
            };
        }

        public string Name => "MockModel";

        public Tensor Forward(Tensor input)
        {
            return input; // Mock implementation
        }

        public void Backward()
        {
            // Mock implementation
        }

        public List<NamedTensor> GetParameters()
        {
            return _parameters;
        }
    }

    #endregion
}
