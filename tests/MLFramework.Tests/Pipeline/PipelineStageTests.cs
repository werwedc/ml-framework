using System;
using System.Linq;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.NN;
using MLFramework.Pipeline;
using Xunit;

namespace MLFramework.Tests.Pipeline
{
    /// <summary>
    /// Simple test module for PipelineStage tests
    /// </summary>
    public class SimpleTestModule : Module
    {
        private readonly Parameter _weight;
        private readonly Parameter? _bias;

        public Parameter Weight => _weight;
        public Parameter? Bias => _bias;

        public SimpleTestModule(int inputSize, int outputSize, bool useBias = true)
            : base("SimpleTestModule")
        {
            var weightData = new float[inputSize * outputSize];
            var biasData = useBias ? new float[outputSize] : null;

            _weight = new Parameter(weightData, new[] { outputSize, inputSize }, "weight", requiresGrad: true);

            if (useBias && biasData != null)
            {
                _bias = new Parameter(biasData, new[] { outputSize }, "bias", requiresGrad: true);
            }
        }

        public override Tensor Forward(Tensor input)
        {
            // Simple forward pass: output = input @ weight.T + bias
            // For simplicity, just return the input (this is a test module)
            return input;
        }

        public override System.Collections.Generic.IEnumerable<Parameter> GetParameters()
        {
            yield return _weight;
            if (_bias != null)
            {
                yield return _bias;
            }
        }

        public override System.Collections.Generic.IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters()
        {
            yield return ("weight", _weight);
            if (_bias != null)
            {
                yield return ("bias", _bias);
            }
        }
    }

    /// <summary>
    /// Unit tests for PipelineStage
    /// </summary>
    public class PipelineStageTests
    {
        [Fact]
        public void Constructor_WithValidInputs_CreatesPipelineStage()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128);
            var rank = 1;
            var totalStages = 4;
            var device = Device.CPU;

            // Act
            var stage = new PipelineStage(module, rank, totalStages, device);

            // Assert
            Assert.NotNull(stage);
            Assert.Equal(rank, stage.Rank);
            Assert.Equal(totalStages, stage.TotalStages);
            Assert.Equal(device, stage.Device);
            Assert.Equal(module, stage.Module);
            Assert.Equal($"PipelineStage_{rank}", stage.Name);
        }

        [Fact]
        public void Constructor_WithRankZero_SetsIsFirstStageToTrue()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128);
            var device = Device.CPU;

            // Act
            var stage = new PipelineStage(module, rank: 0, totalStages: 4, device);

            // Assert
            Assert.True(stage.IsFirstStage);
            Assert.False(stage.IsLastStage);
        }

        [Fact]
        public void Constructor_WithRankEqualToTotalStagesMinusOne_SetsIsLastStageToTrue()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128);
            var device = Device.CPU;

            // Act
            var stage = new PipelineStage(module, rank: 3, totalStages: 4, device);

            // Assert
            Assert.False(stage.IsFirstStage);
            Assert.True(stage.IsLastStage);
        }

        [Fact]
        public void Constructor_WithMiddleRank_SetsBothPropertiesToFalse()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128);
            var device = Device.CPU;

            // Act
            var stage = new PipelineStage(module, rank: 1, totalStages: 4, device);

            // Assert
            Assert.False(stage.IsFirstStage);
            Assert.False(stage.IsLastStage);
        }

        [Fact]
        public void Constructor_WithNullModule_ThrowsArgumentNullException()
        {
            // Arrange
            var device = Device.CPU;

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new PipelineStage(null!, rank: 0, totalStages: 1, device));
        }

        [Fact]
        public void Constructor_WithNullDevice_ThrowsArgumentNullException()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new PipelineStage(module, rank: 0, totalStages: 1, null!));
        }

        [Fact]
        public void Constructor_WithZeroTotalStages_ThrowsArgumentException()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128);
            var device = Device.CPU;

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new PipelineStage(module, rank: 0, totalStages: 0, device));
        }

        [Fact]
        public void Constructor_WithNegativeTotalStages_ThrowsArgumentException()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128);
            var device = Device.CPU;

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new PipelineStage(module, rank: 0, totalStages: -1, device));
        }

        [Fact]
        public void Constructor_WithNegativeRank_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128);
            var device = Device.CPU;

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new PipelineStage(module, rank: -1, totalStages: 4, device));
        }

        [Fact]
        public void Constructor_WithRankEqualToTotalStages_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128);
            var device = Device.CPU;

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new PipelineStage(module, rank: 4, totalStages: 4, device));
        }

        [Fact]
        public void Constructor_WithRankGreaterThanTotalStages_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128);
            var device = Device.CPU;

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new PipelineStage(module, rank: 5, totalStages: 4, device));
        }

        [Fact]
        public void Forward_DelegatesToInnerModule()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128);
            var stage = new PipelineStage(module, rank: 0, totalStages: 1, Device.CPU);
            var input = new Tensor(new float[10 * 64], new[] { 10, 64 });

            // Act
            var output = stage.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(input.Shape, output.Shape); // SimpleTestModule returns input
        }

        [Fact]
        public void Forward_WithNullInput_ThrowsArgumentNullException()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128);
            var stage = new PipelineStage(module, rank: 0, totalStages: 1, Device.CPU);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => stage.Forward(null!));
        }

        [Fact]
        public void GetParameters_ReturnsInnerModuleParameters()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128, useBias: true);
            var stage = new PipelineStage(module, rank: 0, totalStages: 1, Device.CPU);

            // Act
            var parameters = stage.GetParameters();

            // Assert
            Assert.Equal(2, parameters.Count()); // weight + bias
        }

        [Fact]
        public void GetParameters_WithoutBias_ReturnsSingleParameter()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128, useBias: false);
            var stage = new PipelineStage(module, rank: 0, totalStages: 1, Device.CPU);

            // Act
            var parameters = stage.GetParameters();

            // Assert
            Assert.Single(parameters);
        }

        [Fact]
        public void GetNamedParameters_ReturnsInnerModuleNamedParameters()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128, useBias: true);
            var stage = new PipelineStage(module, rank: 0, totalStages: 1, Device.CPU);

            // Act
            var namedParams = stage.GetNamedParameters();

            // Assert
            Assert.Equal(2, namedParams.Count());
            var paramNames = namedParams.Select(p => p.Name).ToList();
            Assert.Contains("weight", paramNames);
            Assert.Contains("bias", paramNames);
        }

        [Fact]
        public void SetRequiresGrad_UpdatesInnerModuleParameters()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128, useBias: true);
            var stage = new PipelineStage(module, rank: 0, totalStages: 1, Device.CPU);

            // Act
            stage.SetRequiresGrad(false);

            // Assert
            Assert.False(module.Weight.RequiresGrad);
            Assert.False(module.Bias!.RequiresGrad);
        }

        [Fact]
        public void GetParameters_ReturnsSameParametersAsInnerModule()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128, useBias: true);
            var stage = new PipelineStage(module, rank: 0, totalStages: 1, Device.CPU);

            // Act
            var stageParams = stage.GetParameters();
            var moduleParams = module.GetParameters();

            // Assert
            Assert.Equal(moduleParams.Count(), stageParams.Count());
        }

        [Fact]
        public void ModuleProperty_ReturnsWrappedModule()
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128);
            var stage = new PipelineStage(module, rank: 0, totalStages: 1, Device.CPU);

            // Act & Assert
            Assert.Same(module, stage.Module);
        }

        [Theory]
        [InlineData(0, 1)]
        [InlineData(1, 2)]
        [InlineData(2, 3)]
        [InlineData(3, 4)]
        [InlineData(0, 8)]
        public void Constructor_WithVariousValidRankAndStages_CreatesPipelineStage(int rank, int totalStages)
        {
            // Arrange
            var module = new SimpleTestModule(inputSize: 64, outputSize: 128);
            var device = Device.CPU;

            // Act
            var stage = new PipelineStage(module, rank, totalStages, device);

            // Assert
            Assert.Equal(rank, stage.Rank);
            Assert.Equal(totalStages, stage.TotalStages);
            Assert.Equal(rank == 0, stage.IsFirstStage);
            Assert.Equal(rank == totalStages - 1, stage.IsLastStage);
        }
    }

    /// <summary>
    /// Unit tests for PipelineConfig
    /// </summary>
    public class PipelineConfigTests
    {
        [Fact]
        public void Constructor_WithDefaultValues_CreatesValidConfig()
        {
            // Act
            var config = new PipelineConfig
            {
                NumStages = 4
            };

            // Assert
            Assert.Equal(4, config.NumStages);
            Assert.Equal(4, config.MicroBatches); // default value
            Assert.Null(config.Devices);
        }

        [Fact]
        public void Validate_WithValidNumStages_Passes()
        {
            // Arrange
            var config = new PipelineConfig
            {
                NumStages = 4,
                MicroBatches = 8
            };

            // Act & Assert (should not throw)
            config.Validate();
        }

        [Fact]
        public void Validate_WithZeroNumStages_ThrowsArgumentException()
        {
            // Arrange
            var config = new PipelineConfig
            {
                NumStages = 0
            };

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => config.Validate());
            Assert.Contains("NumStages", ex.Message);
        }

        [Fact]
        public void Validate_WithNegativeNumStages_ThrowsArgumentException()
        {
            // Arrange
            var config = new PipelineConfig
            {
                NumStages = -1
            };

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => config.Validate());
            Assert.Contains("NumStages", ex.Message);
        }

        [Fact]
        public void Validate_WithZeroMicroBatches_ThrowsArgumentException()
        {
            // Arrange
            var config = new PipelineConfig
            {
                NumStages = 4,
                MicroBatches = 0
            };

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => config.Validate());
            Assert.Contains("MicroBatches", ex.Message);
        }

        [Fact]
        public void Validate_WithNegativeMicroBatches_ThrowsArgumentException()
        {
            // Arrange
            var config = new PipelineConfig
            {
                NumStages = 4,
                MicroBatches = -1
            };

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => config.Validate());
            Assert.Contains("MicroBatches", ex.Message);
        }

        [Fact]
        public void Validate_WithMatchingDevicesLength_Passes()
        {
            // Arrange
            var config = new PipelineConfig
            {
                NumStages = 4,
                MicroBatches = 8,
                Devices = new IDevice[] { Device.CPU, Device.CPU, Device.CPU, Device.CPU }
            };

            // Act & Assert (should not throw)
            config.Validate();
        }

        [Fact]
        public void Validate_WithMismatchedDevicesLength_ThrowsArgumentException()
        {
            // Arrange
            var config = new PipelineConfig
            {
                NumStages = 4,
                MicroBatches = 8,
                Devices = new IDevice[] { Device.CPU, Device.CPU, Device.CPU } // only 3 devices
            };

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => config.Validate());
            Assert.Contains("Devices", ex.Message);
        }

        [Fact]
        public void NumStages_CanBeSetAndGet()
        {
            // Arrange
            var config = new PipelineConfig();

            // Act
            config.NumStages = 8;

            // Assert
            Assert.Equal(8, config.NumStages);
        }

        [Fact]
        public void MicroBatches_CanBeSetAndGet()
        {
            // Arrange
            var config = new PipelineConfig();

            // Act
            config.MicroBatches = 16;

            // Assert
            Assert.Equal(16, config.MicroBatches);
        }

        [Fact]
        public void Devices_CanBeSetAndGet()
        {
            // Arrange
            var config = new PipelineConfig();
            var devices = new IDevice[] { Device.CPU, Device.CPU };

            // Act
            config.Devices = devices;

            // Assert
            Assert.Same(devices, config.Devices);
        }

        [Theory]
        [InlineData(1, 1)]
        [InlineData(2, 4)]
        [InlineData(4, 8)]
        [InlineData(8, 16)]
        [InlineData(16, 32)]
        public void Validate_WithVariousValidValues_Passes(int numStages, int microBatches)
        {
            // Arrange
            var config = new PipelineConfig
            {
                NumStages = numStages,
                MicroBatches = microBatches
            };

            // Act & Assert (should not throw)
            config.Validate();
        }
    }
}
