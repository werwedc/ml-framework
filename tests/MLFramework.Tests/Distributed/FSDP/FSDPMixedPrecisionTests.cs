using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Distributed;
using MLFramework.Distributed.FSDP;
using MLFramework.Tests.Distributed;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;

namespace MLFramework.Tests.Distributed.FSDP
{
    [TestClass]
    public class FSDPMixedPrecisionConfigTests
    {
        [TestMethod]
        public void FSDPMixedPrecisionConfig_DefaultValues_AreCorrect()
        {
            var config = new FSDPMixedPrecisionConfig();

            Assert.IsTrue(config.Enabled);
            Assert.AreEqual(DataType.Float16, config.ForwardDType);
            Assert.AreEqual(DataType.Float32, config.BackwardDType);
            Assert.IsTrue(config.UseLossScaling);
            Assert.AreEqual(2.0f, config.InitialLossScale);
            Assert.AreEqual(1.0f, config.MinLossScale);
            Assert.AreEqual(65536.0f, config.MaxLossScale);
            Assert.AreEqual(2.0f, config.LossScaleGrowthFactor);
            Assert.AreEqual(0.5f, config.LossScaleBackoffFactor);
            Assert.AreEqual(2000, config.LossScaleSteps);
        }

        [TestMethod]
        public void FSDPMixedPrecisionConfig_Enabled_CanBeSet()
        {
            var config = new FSDPMixedPrecisionConfig { Enabled = false };
            Assert.IsFalse(config.Enabled);
        }

        [TestMethod]
        public void FSDPMixedPrecisionConfig_ForwardDType_FP16_CanBeSet()
        {
            var config = new FSDPMixedPrecisionConfig { ForwardDType = DataType.Float16 };
            Assert.AreEqual(DataType.Float16, config.ForwardDType);
        }

        [TestMethod]
        public void FSDPMixedPrecisionConfig_ForwardDType_BF16_CanBeSet()
        {
            var config = new FSDPMixedPrecisionConfig { ForwardDType = DataType.BFloat16 };
            Assert.AreEqual(DataType.BFloat16, config.ForwardDType);
        }

        [TestMethod]
        public void FSDPMixedPrecisionConfig_BackwardDType_FP32_CanBeSet()
        {
            var config = new FSDPMixedPrecisionConfig { BackwardDType = DataType.Float32 };
            Assert.AreEqual(DataType.Float32, config.BackwardDType);
        }

        [TestMethod]
        public void FSDPMixedPrecisionConfig_UseLossScaling_CanBeSet()
        {
            var config = new FSDPMixedPrecisionConfig { UseLossScaling = false };
            Assert.IsFalse(config.UseLossScaling);
        }

        [TestMethod]
        public void FSDPMixedPrecisionConfig_InitialLossScale_CanBeSet()
        {
            var config = new FSDPMixedPrecisionConfig { InitialLossScale = 8.0f };
            Assert.AreEqual(8.0f, config.InitialLossScale);
        }

        [TestMethod]
        public void FSDPMixedPrecisionConfig_MinLossScale_CanBeSet()
        {
            var config = new FSDPMixedPrecisionConfig { MinLossScale = 0.5f };
            Assert.AreEqual(0.5f, config.MinLossScale);
        }

        [TestMethod]
        public void FSDPMixedPrecisionConfig_MaxLossScale_CanBeSet()
        {
            var config = new FSDPMixedPrecisionConfig { MaxLossScale = 32768.0f };
            Assert.AreEqual(32768.0f, config.MaxLossScale);
        }

        [TestMethod]
        public void FSDPMixedPrecisionConfig_LossScaleGrowthFactor_CanBeSet()
        {
            var config = new FSDPMixedPrecisionConfig { LossScaleGrowthFactor = 4.0f };
            Assert.AreEqual(4.0f, config.LossScaleGrowthFactor);
        }

        [TestMethod]
        public void FSDPMixedPrecisionConfig_LossScaleBackoffFactor_CanBeSet()
        {
            var config = new FSDPMixedPrecisionConfig { LossScaleBackoffFactor = 0.25f };
            Assert.AreEqual(0.25f, config.LossScaleBackoffFactor);
        }

        [TestMethod]
        public void FSDPMixedPrecisionConfig_LossScaleSteps_CanBeSet()
        {
            var config = new FSDPMixedPrecisionConfig { LossScaleSteps = 1000 };
            Assert.AreEqual(1000, config.LossScaleSteps);
        }

        [TestMethod]
        public void FSDPMixedPrecisionConfig_Validate_ValidFP16_DoesNotThrow()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                ForwardDType = DataType.Float16,
                BackwardDType = DataType.Float32,
                InitialLossScale = 2.0f,
                MinLossScale = 1.0f,
                MaxLossScale = 65536.0f
            };

            config.Validate(); // Should not throw
        }

        [TestMethod]
        public void FSDPMixedPrecisionConfig_Validate_ValidBF16_DoesNotThrow()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                ForwardDType = DataType.BFloat16,
                BackwardDType = DataType.Float32,
                InitialLossScale = 2.0f,
                MinLossScale = 1.0f,
                MaxLossScale = 65536.0f
            };

            config.Validate(); // Should not throw
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void FSDPMixedPrecisionConfig_Validate_ForwardDTypeFP32_ThrowsArgumentException()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                ForwardDType = DataType.Float32
            };

            config.Validate();
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void FSDPMixedPrecisionConfig_Validate_BackwardDTypeFP16_ThrowsArgumentException()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                BackwardDType = DataType.Float16
            };

            config.Validate();
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void FSDPMixedPrecisionConfig_Validate_InitialLossScaleTooLow_ThrowsArgumentException()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                MinLossScale = 1.0f,
                MaxLossScale = 65536.0f,
                InitialLossScale = 0.5f
            };

            config.Validate();
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void FSDPMixedPrecisionConfig_Validate_InitialLossScaleTooHigh_ThrowsArgumentException()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                MinLossScale = 1.0f,
                MaxLossScale = 65536.0f,
                InitialLossScale = 100000.0f
            };

            config.Validate();
        }

        [TestMethod]
        public void FSDPMixedPrecisionConfig_Validate_BoundaryValues_DoesNotThrow()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                MinLossScale = 1.0f,
                MaxLossScale = 65536.0f,
                InitialLossScale = 1.0f
            };

            config.Validate(); // Should not throw

            config.InitialLossScale = 65536.0f;
            config.Validate(); // Should not throw
        }
    }

    [TestClass]
    public class FSDPMixedPrecisionManagerTests
    {
        private MockProcessGroup _processGroup;
        private MockModel _model;
        private FSDP _fsdp;

        [TestInitialize]
        public void Setup()
        {
            _processGroup = MockProcessGroup.Create(1, 0);
            _model = new MockModel();
            var config = new FSDPConfig();
            _fsdp = new FSDP(_model, config, _processGroup);
        }

        [TestCleanup]
        public void Cleanup()
        {
            _fsdp?.Dispose();
            _processGroup?.Dispose();
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void FSDPMixedPrecisionManager_Constructor_NullConfig_ThrowsArgumentNullException()
        {
            var manager = new FSDPMixedPrecisionManager(null, _fsdp);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void FSDPMixedPrecisionManager_Constructor_NullFSDP_ThrowsArgumentNullException()
        {
            var config = new FSDPMixedPrecisionConfig();
            var manager = new FSDPMixedPrecisionManager(config, null);
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_Constructor_ValidParameters_CreatesInstance()
        {
            var config = new FSDPMixedPrecisionConfig();
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            Assert.IsNotNull(manager);
            Assert.AreEqual(config.InitialLossScale, manager.CurrentLossScale);
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_Constructor_InvalidConfig_ThrowsArgumentException()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                ForwardDType = DataType.Float32 // Invalid
            };

            Assert.ThrowsException<ArgumentException>(() =>
            {
                var manager = new FSDPMixedPrecisionManager(config, _fsdp);
            });
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void FSDPMixedPrecisionManager_ConvertToMixedPrecision_NullTensor_ThrowsArgumentNullException()
        {
            var config = new FSDPMixedPrecisionConfig();
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            manager.ConvertToMixedPrecision(null);
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_ConvertToMixedPrecision_FP32ToFP16_ConvertsCorrectly()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                ForwardDType = DataType.Float16
            };
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            var input = Tensor.Zeros(new[] { 10 }, DataType.Float32);
            for (int i = 0; i < 10; i++)
            {
                input.Data[i] = i * 0.1f;
            }

            var result = manager.ConvertToMixedPrecision(input);

            Assert.IsNotNull(result);
            Assert.AreEqual(DataType.Float16, result.Dtype);
            Assert.AreEqual(input.Size, result.Size);

            // Verify values are preserved (within precision)
            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual(input.Data[i], result.Data[i], 0.001f);
            }
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_ConvertToMixedPrecision_FP32ToBF16_ConvertsCorrectly()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                ForwardDType = DataType.BFloat16
            };
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            var input = Tensor.Zeros(new[] { 10 }, DataType.Float32);
            for (int i = 0; i < 10; i++)
            {
                input.Data[i] = i * 0.1f;
            }

            var result = manager.ConvertToMixedPrecision(input);

            Assert.IsNotNull(result);
            Assert.AreEqual(DataType.BFloat16, result.Dtype);
            Assert.AreEqual(input.Size, result.Size);

            // Verify values are preserved (within precision)
            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual(input.Data[i], result.Data[i], 0.001f);
            }
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_ConvertToMixedPrecision_SameDType_ReturnsSame()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                ForwardDType = DataType.Float16
            };
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            var input = Tensor.Zeros(new[] { 5 }, DataType.Float16);
            var result = manager.ConvertToMixedPrecision(input);

            Assert.AreSame(input, result);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void FSDPMixedPrecisionManager_ConvertGradientToFP32_NullTensor_ThrowsArgumentNullException()
        {
            var config = new FSDPMixedPrecisionConfig();
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            manager.ConvertGradientToFP32(null);
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_ConvertGradientToFP32_FP16ToFP32_ConvertsCorrectly()
        {
            var config = new FSDPMixedPrecisionConfig();
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            var input = Tensor.Zeros(new[] { 10 }, DataType.Float16);
            for (int i = 0; i < 10; i++)
            {
                input.Data[i] = i * 0.01f;
            }

            var result = manager.ConvertGradientToFP32(input);

            Assert.IsNotNull(result);
            Assert.AreEqual(DataType.Float32, result.Dtype);
            Assert.AreEqual(input.Size, result.Size);

            // Verify values are preserved
            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual(input.Data[i], result.Data[i], 0.0001f);
            }
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_ScaleLoss_WithoutLossScaling_ReturnsOriginal()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                UseLossScaling = false
            };
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            var loss = Tensor.Zeros(new[] { 1 }, DataType.Float32);
            loss.Data[0] = 1.5f;

            var result = manager.ScaleLoss(loss);

            Assert.AreSame(loss, result);
            Assert.AreEqual(1.5f, result.Data[0]);
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_ScaleLoss_WithLossScaling_ScalesCorrectly()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                UseLossScaling = true,
                InitialLossScale = 4.0f
            };
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            var loss = Tensor.Zeros(new[] { 1 }, DataType.Float32);
            loss.Data[0] = 1.5f;

            var result = manager.ScaleLoss(loss);

            Assert.AreNotSame(loss, result);
            Assert.AreEqual(6.0f, result.Data[0], 0.001f); // 1.5 * 4
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_CheckOverflow_WithoutLossScaling_ReturnsFalse()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                UseLossScaling = false
            };
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            var gradients = new Dictionary<string, Tensor>();
            var grad = Tensor.Zeros(new[] { 5 }, DataType.Float32);
            gradients["param1"] = grad;

            var overflow = manager.CheckOverflow(gradients);

            Assert.IsFalse(overflow);
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_CheckOverflow_NoOverflow_ReturnsFalse()
        {
            var config = new FSDPMixedPrecisionConfig();
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            var gradients = new Dictionary<string, Tensor>();
            var grad = Tensor.Zeros(new[] { 5 }, DataType.Float32);
            for (int i = 0; i < 5; i++)
            {
                grad.Data[i] = i * 0.1f;
            }
            gradients["param1"] = grad;

            var overflow = manager.CheckOverflow(gradients);

            Assert.IsFalse(overflow);
            Assert.AreEqual(config.InitialLossScale, manager.CurrentLossScale);
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_CheckOverflow_NaN_DetectsOverflow()
        {
            var config = new FSDPMixedPrecisionConfig();
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            var gradients = new Dictionary<string, Tensor>();
            var grad = Tensor.Zeros(new[] { 5 }, DataType.Float32);
            grad.Data[2] = float.NaN;
            gradients["param1"] = grad;

            var overflow = manager.CheckOverflow(gradients);

            Assert.IsTrue(overflow);
            Assert.AreEqual(
                config.MinLossScale,
                manager.CurrentLossScale,
                0.001f
            );
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_CheckOverflow_PositiveInfinity_DetectsOverflow()
        {
            var config = new FSDPMixedPrecisionConfig();
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            var gradients = new Dictionary<string, Tensor>();
            var grad = Tensor.Zeros(new[] { 5 }, DataType.Float32);
            grad.Data[3] = float.PositiveInfinity;
            gradients["param1"] = grad;

            var overflow = manager.CheckOverflow(gradients);

            Assert.IsTrue(overflow);
            Assert.AreEqual(
                config.MinLossScale,
                manager.CurrentLossScale,
                0.001f
            );
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_CheckOverflow_NegativeInfinity_DetectsOverflow()
        {
            var config = new FSDPMixedPrecisionConfig();
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            var gradients = new Dictionary<string, Tensor>();
            var grad = Tensor.Zeros(new[] { 5 }, DataType.Float32);
            grad.Data[1] = float.NegativeInfinity;
            gradients["param1"] = grad;

            var overflow = manager.CheckOverflow(gradients);

            Assert.IsTrue(overflow);
            Assert.AreEqual(
                config.MinLossScale,
                manager.CurrentLossScale,
                0.001f
            );
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_CheckOverflow_BackoffLossScale_ReducesCorrectly()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                InitialLossScale = 8.0f,
                MinLossScale = 1.0f,
                LossScaleBackoffFactor = 0.5f
            };
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            var gradients = new Dictionary<string, Tensor>();
            var grad = Tensor.Zeros(new[] { 1 }, DataType.Float32);
            grad.Data[0] = float.NaN;
            gradients["param1"] = grad;

            manager.CheckOverflow(gradients);

            Assert.AreEqual(4.0f, manager.CurrentLossScale, 0.001f);
            Assert.AreEqual(0, manager._stepsSinceOverflow);
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_CheckOverflow_StepsWithoutOverflow_IncreasesLossScale()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                InitialLossScale = 2.0f,
                MaxLossScale = 16.0f,
                LossScaleSteps = 3,
                LossScaleGrowthFactor = 2.0f
            };
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            var gradients = new Dictionary<string, Tensor>();
            var grad = Tensor.Zeros(new[] { 1 }, DataType.Float32);
            grad.Data[0] = 1.0f;
            gradients["param1"] = grad;

            // Check overflow multiple times without overflow
            manager.CheckOverflow(gradients); // Step 1
            manager.CheckOverflow(gradients); // Step 2
            manager.CheckOverflow(gradients); // Step 3 - should trigger increase

            Assert.AreEqual(4.0f, manager.CurrentLossScale, 0.001f);
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_ResetLossScale_ResetsToInitial()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                InitialLossScale = 4.0f
            };
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            // Change loss scale
            manager.CheckOverflow(new Dictionary<string, Tensor>()); // This shouldn't change it, but let's ensure

            manager.ResetLossScale();

            Assert.AreEqual(config.InitialLossScale, manager.CurrentLossScale);
        }

        [TestMethod]
        public void FSDPMixedPrecisionManager_Dispose_CanBeCalledMultipleTimes()
        {
            var config = new FSDPMixedPrecisionConfig();
            var manager = new FSDPMixedPrecisionManager(config, _fsdp);

            manager.Dispose();
            manager.Dispose(); // Should not throw
        }
    }

    [TestClass]
    public class FSDPAmpIntegrationTests
    {
        private MockProcessGroup _processGroup;
        private MockModel _model;
        private FSDP _fsdp;

        [TestInitialize]
        public void Setup()
        {
            _processGroup = MockProcessGroup.Create(1, 0);
            _model = new MockModel();
            var config = new FSDPConfig();
            _fsdp = new FSDP(_model, config, _processGroup);
        }

        [TestCleanup]
        public void Cleanup()
        {
            _fsdp?.Dispose();
            _processGroup?.Dispose();
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void FSDPAmpIntegration_Constructor_NullFSDP_ThrowsArgumentNullException()
        {
            var integration = new FSDPAmpIntegration(null);
        }

        [TestMethod]
        public void FSDPAmpIntegration_Constructor_DefaultConfig_CreatesInstance()
        {
            var integration = new FSDPAmpIntegration(_fsdp);

            Assert.IsNotNull(integration);
            Assert.IsNotNull(integration.Manager);
        }

        [TestMethod]
        public void FSDPAmpIntegration_Constructor_CustomConfig_CreatesInstance()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                ForwardDType = DataType.BFloat16,
                InitialLossScale = 4.0f
            };
            var integration = new FSDPAmpIntegration(_fsdp, config);

            Assert.IsNotNull(integration);
            Assert.AreEqual(4.0f, integration.Manager.CurrentLossScale);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void FSDPAmpIntegration_ApplyMixedPrecision_NullShardingUnit_ThrowsArgumentNullException()
        {
            var integration = new FSDPAmpIntegration(_fsdp);
            integration.ApplyMixedPrecision(null);
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void FSDPAmpIntegration_ApplyMixedPrecision_NoGatheredParameter_ThrowsInvalidOperationException()
        {
            var integration = new FSDPAmpIntegration(_fsdp);
            var unit = new FSDPShardingUnit("param", Tensor.Zeros(new[] { 10 }, DataType.Float32), _processGroup);

            integration.ApplyMixedPrecision(unit);
        }

        [TestMethod]
        public void FSDPAmpIntegration_ApplyMixedPrecision_ConvertsToFP16()
        {
            var integration = new FSDPAmpIntegration(_fsdp);
            var param = Tensor.Zeros(new[] { 10 }, DataType.Float32);
            for (int i = 0; i < 10; i++)
            {
                param.Data[i] = i * 0.1f;
            }
            var unit = new FSDPShardingUnit("param", param, _processGroup);
            unit.GatheredParameter = param.Clone();

            integration.ApplyMixedPrecision(unit);

            Assert.AreEqual(DataType.Float16, unit.GatheredParameter.Dtype);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void FSDPAmpIntegration_ApplyGradientFP32_NullShardingUnit_ThrowsArgumentNullException()
        {
            var integration = new FSDPAmpIntegration(_fsdp);
            integration.ApplyGradientFP32(null);
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void FSDPAmpIntegration_ApplyGradientFP32_NoGradient_ThrowsInvalidOperationException()
        {
            var integration = new FSDPAmpIntegration(_fsdp);
            var param = Tensor.Zeros(new[] { 10 }, DataType.Float32);
            var unit = new FSDPShardingUnit("param", param, _processGroup);

            integration.ApplyGradientFP32(unit);
        }

        [TestMethod]
        public void FSDPAmpIntegration_ApplyGradientFP32_ConvertsToFP32()
        {
            var integration = new FSDPAmpIntegration(_fsdp);
            var param = Tensor.Zeros(new[] { 10 }, DataType.Float32);
            var unit = new FSDPShardingUnit("param", param, _processGroup);
            unit.LocalGradient = Tensor.Zeros(new[] { 10 }, DataType.Float16);

            integration.ApplyGradientFP32(unit);

            Assert.AreEqual(DataType.Float32, unit.LocalGradient.Dtype);
        }

        [TestMethod]
        public void FSDPAmpIntegration_ScaleLoss_ScalesCorrectly()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                InitialLossScale = 3.0f
            };
            var integration = new FSDPAmpIntegration(_fsdp, config);
            var loss = Tensor.Zeros(new[] { 1 }, DataType.Float32);
            loss.Data[0] = 2.0f;

            var result = integration.ScaleLoss(loss);

            Assert.AreEqual(6.0f, result.Data[0], 0.001f);
        }

        [TestMethod]
        public void FSDPAmpIntegration_CheckOverflow_PassesThroughToManager()
        {
            var integration = new FSDPAmpIntegration(_fsdp);
            var gradients = new Dictionary<string, Tensor>();
            var grad = Tensor.Zeros(new[] { 1 }, DataType.Float32);
            gradients["param1"] = grad;

            var overflow = integration.CheckOverflow(gradients);

            Assert.IsFalse(overflow);
        }

        [TestMethod]
        public void FSDPAmpIntegration_Manager_ReturnsCorrectInstance()
        {
            var config = new FSDPMixedPrecisionConfig();
            var integration = new FSDPAmpIntegration(_fsdp, config);

            var manager = integration.Manager;

            Assert.IsNotNull(manager);
            Assert.AreEqual(config.InitialLossScale, manager.CurrentLossScale);
        }

        [TestMethod]
        public void FSDPAmpIntegration_Dispose_CanBeCalledMultipleTimes()
        {
            var integration = new FSDPAmpIntegration(_fsdp);

            integration.Dispose();
            integration.Dispose(); // Should not throw
        }

        [TestMethod]
        public void FSDPAmpIntegration_EndToEnd_Workflow_Succeeds()
        {
            var config = new FSDPMixedPrecisionConfig
            {
                ForwardDType = DataType.Float16,
                InitialLossScale = 2.0f
            };
            var integration = new FSDPAmpIntegration(_fsdp, config);

            // Create parameter
            var param = Tensor.Zeros(new[] { 5 }, DataType.Float32);
            for (int i = 0; i < 5; i++)
            {
                param.Data[i] = (i + 1) * 0.1f;
            }
            var unit = new FSDPShardingUnit("param", param, _processGroup);
            unit.GatheredParameter = param.Clone();

            // Apply mixed precision
            integration.ApplyMixedPrecision(unit);
            Assert.AreEqual(DataType.Float16, unit.GatheredParameter.Dtype);

            // Compute gradient (simulated)
            unit.LocalGradient = Tensor.Zeros(new[] { 5 }, DataType.Float16);
            for (int i = 0; i < 5; i++)
            {
                unit.LocalGradient.Data[i] = (i + 1) * 0.01f;
            }

            // Convert to FP32
            integration.ApplyGradientFP32(unit);
            Assert.AreEqual(DataType.Float32, unit.LocalGradient.Dtype);

            // Scale loss
            var loss = Tensor.Zeros(new[] { 1 }, DataType.Float32);
            loss.Data[0] = 1.0f;
            var scaledLoss = integration.ScaleLoss(loss);
            Assert.AreEqual(2.0f, scaledLoss.Data[0], 0.001f);

            // Check overflow
            var gradients = new Dictionary<string, Tensor>
            {
                { "param", unit.LocalGradient }
            };
            var overflow = integration.CheckOverflow(gradients);
            Assert.IsFalse(overflow);
        }
    }

    #region Mock Classes

    /// <summary>
    /// Mock model for testing.
    /// </summary>
    public class MockModel : IModel
    {
        private readonly List<NamedTensor> _parameters;

        public string Name => "MockModel";

        public MockModel()
        {
            _parameters = new List<NamedTensor>();
            // Add some default parameters
            var weight = Tensor.Zeros(new[] { 10, 10 }, DataType.Float32);
            var bias = Tensor.Zeros(new[] { 10 }, DataType.Float32);
            _parameters.Add(new NamedTensor("weight", weight));
            _parameters.Add(new NamedTensor("bias", bias));
        }

        public Tensor Forward(Tensor input)
        {
            return input;
        }

        public void Backward()
        {
            // No-op for mock
        }

        public List<NamedTensor> GetParameters()
        {
            return _parameters;
        }
    }

    #endregion
}
