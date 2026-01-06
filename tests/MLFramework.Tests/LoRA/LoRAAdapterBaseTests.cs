using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.LoRA;
using MLFramework.Modules;

namespace MLFramework.Tests.LoRA
{
    // Mock module for testing
    public class MockModule : IModule
    {
        public IEnumerable<Tensor> Parameters { get; set; } = Enumerable.Empty<Tensor>();
        public string ModuleType => "MockModule";
        public bool IsTraining { get; set; }

        public Tensor Forward(Tensor input)
        {
            return input;
        }

        public void ApplyToParameters(Action<Tensor> action)
        {
            foreach (var param in Parameters)
            {
                action(param);
            }
        }

        public void SetRequiresGrad(bool requiresGrad)
        {
            foreach (var param in Parameters)
            {
                param.RequiresGrad = requiresGrad;
            }
        }
    }

    // Mock LoRA adapter for testing the base class
    public class MockLoRAAdapter : LoRAAdapterBase
    {
        public MockLoRAAdapter(IModule baseLayer, int rank, float alpha)
            : base(baseLayer, rank, alpha)
        {
        }

        public override void FreezeBaseLayer()
        {
            _isBaseLayerFrozen = true;
        }

        public override void UnfreezeBaseLayer()
        {
            _isBaseLayerFrozen = false;
        }

        public override IEnumerable<Tensor> TrainableParameters => Enumerable.Empty<Tensor>();

        public override IEnumerable<Tensor> FrozenParameters => Enumerable.Empty<Tensor>();

        public override void MergeAdapter()
        {
            // Mock implementation
        }

        public override void ResetBaseLayer()
        {
            // Mock implementation
        }

        public override (Tensor? MatrixA, Tensor? MatrixB) GetAdapterWeights()
        {
            return (null, null);
        }

        public override void SetAdapterWeights(Tensor? matrixA, Tensor? matrixB)
        {
            // Mock implementation
        }
    }

    public class LoRAAdapterBaseTests
    {
        [Fact]
        public void Constructor_WithValidParameters_SetsPropertiesCorrectly()
        {
            // Arrange
            var baseLayer = new MockModule();
            int rank = 16;
            float alpha = 32.0f;

            // Act
            var adapter = new MockLoRAAdapter(baseLayer, rank, alpha);

            // Assert
            Assert.Equal(baseLayer, adapter.BaseLayer);
            Assert.Equal(rank, adapter.Rank);
            Assert.Equal(alpha / rank, adapter.ScalingFactor);
        }

        [Fact]
        public void Constructor_WithNullBaseLayer_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new MockLoRAAdapter(null!, 8, 16.0f));
        }

        [Theory]
        [InlineData(8, 16.0f, 2.0f)]
        [InlineData(4, 8.0f, 2.0f)]
        [InlineData(16, 32.0f, 2.0f)]
        [InlineData(8, 32.0f, 4.0f)]
        public void ScalingFactor_IsCalculatedCorrectly(int rank, float alpha, float expectedScalingFactor)
        {
            // Arrange
            var baseLayer = new MockModule();

            // Act
            var adapter = new MockLoRAAdapter(baseLayer, rank, alpha);

            // Assert
            Assert.Equal(expectedScalingFactor, adapter.ScalingFactor);
        }

        [Fact]
        public void IsEnabled_DefaultValue_IsTrue()
        {
            // Arrange
            var baseLayer = new MockModule();

            // Act
            var adapter = new MockLoRAAdapter(baseLayer, 8, 16.0f);

            // Assert
            Assert.True(adapter.IsEnabled);
        }

        [Fact]
        public void IsEnabled_CanBeSetToFalse()
        {
            // Arrange
            var baseLayer = new MockModule();
            var adapter = new MockLoRAAdapter(baseLayer, 8, 16.0f);

            // Act
            adapter.IsEnabled = false;

            // Assert
            Assert.False(adapter.IsEnabled);
        }

        [Fact]
        public void IsEnabled_CanBeSetBackToTrue()
        {
            // Arrange
            var baseLayer = new MockModule();
            var adapter = new MockLoRAAdapter(baseLayer, 8, 16.0f);
            adapter.IsEnabled = false;

            // Act
            adapter.IsEnabled = true;

            // Assert
            Assert.True(adapter.IsEnabled);
        }

        [Fact]
        public void BaseLayer_WrapsProvidedModule()
        {
            // Arrange
            var baseLayer = new MockModule();

            // Act
            var adapter = new MockLoRAAdapter(baseLayer, 8, 16.0f);

            // Assert
            Assert.Same(baseLayer, adapter.BaseLayer);
            Assert.Equal("MockModule", adapter.BaseLayer.ModuleType);
        }
    }
}
