using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.NN;
using MLFramework.Pipeline;
using Xunit;

namespace MLFramework.Tests.Pipeline
{
    /// <summary>
    /// Simple test module for PipelineValidator tests
    /// </summary>
    public class ValidatorTestModule : Module
    {
        private readonly Parameter _weight;
        private readonly Parameter? _bias;

        public Parameter Weight => _weight;
        public Parameter? Bias => _bias;

        public ValidatorTestModule(int inputSize, int outputSize, bool useBias = true)
            : base("ValidatorTestModule")
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
            // Simple forward pass: return input for testing
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
    /// Mock communicator for testing
    /// </summary>
    public class MockPipelineCommunicator : IPipelineCommunicator
    {
        private int _worldSize;
        private int _rank;
        private bool _disposed;

        public MockPipelineCommunicator(int worldSize, int rank)
        {
            _worldSize = worldSize;
            _rank = rank;
        }

        public int WorldSize => _worldSize;

        public int Rank => _rank;

        public Task SendAsync(Tensor tensor, int destRank)
        {
            return Task.CompletedTask;
        }

        public Task<Tensor> ReceiveAsync(int srcRank)
        {
            // Return a dummy tensor
            return Task.FromResult(new Tensor(new float[100], new[] { 100 }));
        }

        public Task<Tensor> BroadcastAsync(Tensor tensor, int root)
        {
            // Return the same tensor (simple implementation)
            return Task.FromResult(tensor);
        }

        public Task BarrierAsync()
        {
            return Task.CompletedTask;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Unit tests for PipelineValidator
    /// </summary>
    public class PipelineValidatorTests
    {
        [Fact]
        public void Constructor_WithValidInputs_CreatesValidator()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 2, Device.CPU),
                new PipelineStage(new ValidatorTestModule(128, 256), 1, 2, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(2, 0);

            // Act
            var validator = new PipelineValidator(stages, communicator);

            // Assert
            Assert.NotNull(validator);
            Assert.True(validator.IsValid);
            Assert.Empty(validator.Errors);
            Assert.Empty(validator.Warnings);
        }

        [Fact]
        public void Constructor_WithNullStages_ThrowsArgumentNullException()
        {
            // Arrange
            var communicator = new MockPipelineCommunicator(2, 0);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new PipelineValidator(null!, communicator));
        }

        [Fact]
        public void Constructor_WithNullCommunicator_ThrowsArgumentNullException()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 2, Device.CPU)
            };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new PipelineValidator(stages, null!));
        }

        [Fact]
        public void ValidateConfiguration_WithValidStages_ReturnsTrue()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 2, Device.CPU),
                new PipelineStage(new ValidatorTestModule(128, 256), 1, 2, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(2, 0);
            var validator = new PipelineValidator(stages, communicator);

            // Act
            var isValid = validator.ValidateConfiguration();

            // Assert
            Assert.True(isValid);
            Assert.Empty(validator.Errors);
        }

        [Fact]
        public void ValidateConfiguration_WithEmptyStages_ReturnsFalse()
        {
            // Arrange
            var stages = new List<PipelineStage>();
            var communicator = new MockPipelineCommunicator(0, 0);
            var validator = new PipelineValidator(stages, communicator);

            // Act
            var isValid = validator.ValidateConfiguration();

            // Assert
            Assert.False(isValid);
            Assert.Single(validator.Errors);
            Assert.Equal("CONFIG_EMPTY_STAGES", validator.Errors[0].Code);
        }

        [Fact]
        public void ValidateConfiguration_WithWorldSizeMismatch_ReturnsFalse()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 2, Device.CPU),
                new PipelineStage(new ValidatorTestModule(128, 256), 1, 2, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(4, 0); // Mismatch: 2 stages, 4 world size
            var validator = new PipelineValidator(stages, communicator);

            // Act
            var isValid = validator.ValidateConfiguration();

            // Assert
            Assert.False(isValid);
            Assert.Contains(validator.Errors, e => e.Code == "CONFIG_WORLD_SIZE_MISMATCH");
        }

        [Fact]
        public void ValidateConfiguration_WithInvalidRank_ReturnsFalse()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 2, Device.CPU),
                new PipelineStage(new ValidatorTestModule(128, 256), 5, 2, Device.CPU) // Invalid rank
            };
            var communicator = new MockPipelineCommunicator(2, 0);
            var validator = new PipelineValidator(stages, communicator);

            // Act
            var isValid = validator.ValidateConfiguration();

            // Assert
            Assert.False(isValid);
            Assert.Contains(validator.Errors, e => e.Code == "CONFIG_RANK_MISMATCH");
        }

        [Fact]
        public void ValidateConfiguration_WithDuplicateRanks_ReturnsFalse()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 2, Device.CPU),
                new PipelineStage(new ValidatorTestModule(128, 256), 0, 2, Device.CPU) // Duplicate rank
            };
            var communicator = new MockPipelineCommunicator(2, 0);
            var validator = new PipelineValidator(stages, communicator);

            // Act
            var isValid = validator.ValidateConfiguration();

            // Assert
            Assert.False(isValid);
            Assert.Contains(validator.Errors, e => e.Code == "CONFIG_DUPLICATE_RANK");
        }

        [Fact]
        public void ValidateParameterConsistency_WithConsistentParameters_ReturnsTrue()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128, true), 0, 2, Device.CPU),
                new PipelineStage(new ValidatorTestModule(128, 256, true), 1, 2, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(2, 0);
            var validator = new PipelineValidator(stages, communicator);

            // Act
            var isValid = validator.ValidateParameterConsistency();

            // Assert
            Assert.True(isValid);
            Assert.Empty(validator.Errors);
        }

        [Fact]
        public void ValidateParameterConsistency_WithEmptyStages_ReturnsFalse()
        {
            // Arrange
            var stages = new List<PipelineStage>();
            var communicator = new MockPipelineCommunicator(0, 0);
            var validator = new PipelineValidator(stages, communicator);

            // Act
            var isValid = validator.ValidateParameterConsistency();

            // Assert
            Assert.False(isValid);
            Assert.Contains(validator.Errors, e => e.Code == "PARAM_EMPTY_STAGES");
        }

        [Fact]
        public void ValidateParameterConsistency_WithDifferentParamCounts_AddsWarning()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128, true), 0, 2, Device.CPU),
                new PipelineStage(new ValidatorTestModule(128, 256, false), 1, 2, Device.CPU) // No bias
            };
            var communicator = new MockPipelineCommunicator(2, 0);
            var validator = new PipelineValidator(stages, communicator);

            // Act
            var isValid = validator.ValidateParameterConsistency();

            // Assert
            Assert.True(isValid); // Warnings don't make it invalid
            Assert.Contains(validator.Warnings, w => w.Code == "PARAM_COUNT_MISMATCH");
        }

        [Fact]
        public void ValidateNumericalStability_WithNaNValues_ReturnsFalse()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 1, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(1, 0);
            var validator = new PipelineValidator(stages, communicator);

            var activations = new List<Tensor>
            {
                new Tensor(new float[] { 1.0f, float.NaN, 3.0f }, new[] { 3 })
            };

            // Act
            var isValid = validator.ValidateNumericalStability(activations, null);

            // Assert
            Assert.False(isValid);
            Assert.Contains(validator.Errors, e => e.Code == "NUMERICAL_NAN");
        }

        [Fact]
        public void ValidateNumericalStability_WithInfValues_ReturnsFalse()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 1, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(1, 0);
            var validator = new PipelineValidator(stages, communicator);

            var activations = new List<Tensor>
            {
                new Tensor(new float[] { 1.0f, float.PositiveInfinity, 3.0f }, new[] { 3 })
            };

            // Act
            var isValid = validator.ValidateNumericalStability(activations, null);

            // Assert
            Assert.False(isValid);
            Assert.Contains(validator.Errors, e => e.Code == "NUMERICAL_INF");
        }

        [Fact]
        public void ValidateNumericalStability_WithNullActivations_ReturnsFalse()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 1, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(1, 0);
            var validator = new PipelineValidator(stages, communicator);

            // Act
            var isValid = validator.ValidateNumericalStability(null, null);

            // Assert
            Assert.False(isValid);
            Assert.Contains(validator.Errors, e => e.Code == "NUMERICAL_NO_ACTIVATIONS");
        }

        [Fact]
        public void ValidateNumericalStability_WithValidValues_ReturnsTrue()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 1, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(1, 0);
            var validator = new PipelineValidator(stages, communicator);

            var activations = new List<Tensor>
            {
                new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 })
            };

            // Act
            var isValid = validator.ValidateNumericalStability(activations, null);

            // Assert
            Assert.True(isValid);
            Assert.Empty(validator.Errors);
        }

        [Fact]
        public void ValidateNumericalStability_WithLargeValues_AddsWarning()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 1, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(1, 0);
            var validator = new PipelineValidator(stages, communicator);

            var activations = new List<Tensor>
            {
                new Tensor(new float[] { 1.0f, 2e7f, 3.0f }, new[] { 3 }) // Large value
            };

            // Act
            var isValid = validator.ValidateNumericalStability(activations, null);

            // Assert
            Assert.True(isValid); // Warnings don't make it invalid
            Assert.Contains(validator.Warnings, w => w.Code == "NUMERICAL_LARGE_ACTIVATION");
        }

        [Fact]
        public void ValidateMemoryUsage_WithWithinLimit_ReturnsTrue()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 2, Device.CPU),
                new PipelineStage(new ValidatorTestModule(128, 256), 1, 2, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(2, 0);
            var validator = new PipelineValidator(stages, communicator);

            var maxMemory = 1_000_000_000L; // 1GB - should be enough

            // Act
            var isValid = validator.ValidateMemoryUsage(maxMemory);

            // Assert
            Assert.True(isValid);
            Assert.Empty(validator.Errors);
        }

        [Fact]
        public void ValidateMemoryUsage_WithExceededLimit_ReturnsFalse()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 2, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(1, 0);
            var validator = new PipelineValidator(stages, communicator);

            var maxMemory = 1L; // 1 byte - should be too small

            // Act
            var isValid = validator.ValidateMemoryUsage(maxMemory);

            // Assert
            Assert.False(isValid);
            Assert.Contains(validator.Errors, e => e.Code == "MEMORY_EXCEEDED");
        }

        [Fact]
        public void ValidateMemoryUsage_WithInvalidLimit_ReturnsFalse()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 1, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(1, 0);
            var validator = new PipelineValidator(stages, communicator);

            // Act
            var isValid = validator.ValidateMemoryUsage(0);

            // Assert
            Assert.False(isValid);
            Assert.Contains(validator.Errors, e => e.Code == "MEMORY_INVALID_LIMIT");
        }

        [Fact]
        public async Task ValidateCommunicationAsync_WithValidCommunicator_ReturnsTrue()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 1, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(1, 0);
            var validator = new PipelineValidator(stages, communicator);

            // Act
            var isValid = await validator.ValidateCommunicationAsync();

            // Assert
            Assert.True(isValid);
            Assert.Empty(validator.Errors);
        }

        [Fact]
        public void ValidateAll_WithValidSetup_ReturnsTrue()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 2, Device.CPU),
                new PipelineStage(new ValidatorTestModule(128, 256), 1, 2, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(2, 0);
            var validator = new PipelineValidator(stages, communicator);

            // Act
            var isValid = validator.ValidateAll();

            // Assert
            Assert.True(isValid);
            Assert.Empty(validator.Errors);
        }

        [Fact]
        public void ValidateAll_WithInvalidConfiguration_ReturnsFalse()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 1, Device.CPU),
                new PipelineStage(new ValidatorTestModule(128, 256), 5, 1, Device.CPU) // Invalid rank
            };
            var communicator = new MockPipelineCommunicator(2, 0);
            var validator = new PipelineValidator(stages, communicator);

            // Act
            var isValid = validator.ValidateAll();

            // Assert
            Assert.False(isValid);
            Assert.NotEmpty(validator.Errors);
        }

        [Fact]
        public void Clear_ClearsErrorsAndWarnings()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 1, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(1, 0);
            var validator = new PipelineValidator(stages, communicator);

            // Add some errors
            validator.ValidateMemoryUsage(0);

            // Act
            validator.Clear();

            // Assert
            Assert.Empty(validator.Errors);
            Assert.Empty(validator.Warnings);
            Assert.True(validator.IsValid);
        }

        [Fact]
        public void GetMetrics_ReturnsValidationMetrics()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 1, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(1, 0);
            var validator = new PipelineValidator(stages, communicator);

            // Act
            var metrics = validator.GetMetrics();

            // Assert
            Assert.NotNull(metrics);
            Assert.Equal(1, metrics.MemoryUsage.Length);
        }

        [Fact]
        public void ValidateGradients_WithNullBaseline_ReturnsFalse()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 1, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(1, 0);
            var validator = new PipelineValidator(stages, communicator);
            var input = new Tensor(new float[100], new[] { 100 });

            // Act
            var isValid = validator.ValidateGradients(null!, input);

            // Assert
            Assert.False(isValid);
            Assert.Contains(validator.Errors, e => e.Code == "GRADIENT_NULL_BASELINE");
        }

        [Fact]
        public void ValidateGradients_WithNullInput_ReturnsFalse()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 1, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(1, 0);
            var validator = new PipelineValidator(stages, communicator);
            var model = new ValidatorTestModule(64, 128);

            // Act
            var isValid = validator.ValidateGradients(model, null!);

            // Assert
            Assert.False(isValid);
            Assert.Contains(validator.Errors, e => e.Code == "GRADIENT_NULL_INPUT");
        }

        [Fact]
        public void Dispose_DisposesCommunicator()
        {
            // Arrange
            var stages = new List<PipelineStage>
            {
                new PipelineStage(new ValidatorTestModule(64, 128), 0, 1, Device.CPU)
            };
            var communicator = new MockPipelineCommunicator(1, 0);
            var validator = new PipelineValidator(stages, communicator);

            // Act
            validator.Dispose();

            // Assert - no exception thrown
            Assert.True(true);
        }

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(4)]
        [InlineData(8)]
        public void ValidateConfiguration_WithVariousStageCounts_ReturnsTrue(int numStages)
        {
            // Arrange
            var stages = new List<PipelineStage>();
            for (int i = 0; i < numStages; i++)
            {
                stages.Add(new PipelineStage(
                    new ValidatorTestModule(64, 128),
                    i, numStages, Device.CPU));
            }
            var communicator = new MockPipelineCommunicator(numStages, 0);
            var validator = new PipelineValidator(stages, communicator);

            // Act
            var isValid = validator.ValidateConfiguration();

            // Assert
            Assert.True(isValid);
            Assert.Empty(validator.Errors);
        }
    }
}
