using RitterFramework.Core.Tensor;
using MLFramework.Modules;
using Xunit;

namespace MLFramework.Tests.Modules
{
    /// <summary>
    /// Unit tests for Linear layer
    /// </summary>
    public class LinearTests
    {
        [Fact]
        public void Constructor_WithValidParameters_CreatesLinearLayer()
        {
            // Act
            var linear = new Linear(inFeatures: 64, outFeatures: 128);

            // Assert
            Assert.NotNull(linear);
            Assert.Equal(64, linear.InFeatures);
            Assert.Equal(128, linear.OutFeatures);
            Assert.Equal("Linear", linear.ModuleType);
            Assert.NotNull(linear.Weight);
            Assert.Equal(new[] { 128, 64 }, linear.Weight.Shape);
            Assert.NotNull(linear.Bias);
            Assert.Equal(new[] { 128 }, linear.Bias.Shape);
        }

        [Fact]
        public void Constructor_WithoutBias_CreatesLinearLayerWithoutBias()
        {
            // Act
            var linear = new Linear(inFeatures: 64, outFeatures: 128, useBias: false);

            // Assert
            Assert.NotNull(linear);
            Assert.Null(linear.Bias);
        }

        [Fact]
        public void Constructor_WithNegativeInFeatures_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new Linear(inFeatures: -1, outFeatures: 128));
        }

        [Fact]
        public void Constructor_WithZeroInFeatures_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new Linear(inFeatures: 0, outFeatures: 128));
        }

        [Fact]
        public void Constructor_WithNegativeOutFeatures_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new Linear(inFeatures: 64, outFeatures: -1));
        }

        [Fact]
        public void Constructor_WithZeroOutFeatures_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new Linear(inFeatures: 64, outFeatures: 0));
        }

        [Fact]
        public void Forward_ProducesCorrectOutputShape()
        {
            // Arrange
            var linear = new Linear(inFeatures: 64, outFeatures: 128);
            var input = new Tensor(new float[10 * 64], new[] { 10, 64 }); // batch=10, in=64

            // Act
            var output = linear.Forward(input);

            // Assert
            Assert.Equal(new[] { 10, 128 }, output.Shape);
        }

        [Fact]
        public void Forward_With2DInput_ProducesCorrectOutput()
        {
            // Arrange
            var linear = new Linear(inFeatures: 4, outFeatures: 8);
            var inputData = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var input = new Tensor(inputData, new[] { 2, 4 }); // batch=2, in=4

            // Act
            var output = linear.Forward(input);

            // Assert
            Assert.Equal(new[] { 2, 8 }, output.Shape);
            Assert.NotNull(output);
        }

        [Fact]
        public void Forward_WithNullInput_ThrowsArgumentNullException()
        {
            // Arrange
            var linear = new Linear(inFeatures: 64, outFeatures: 128);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => linear.Forward(null!));
        }

        [Fact]
        public void Forward_WithWrongInputDimension_ThrowsArgumentException()
        {
            // Arrange
            var linear = new Linear(inFeatures: 64, outFeatures: 128);
            var input = new Tensor(new float[10 * 32], new[] { 10, 32 }); // Wrong input dimension

            // Act & Assert
            Assert.Throws<ArgumentException>(() => linear.Forward(input));
        }

        [Fact]
        public void Parameters_ReturnsWeight()
        {
            // Arrange
            var linear = new Linear(inFeatures: 64, outFeatures: 128, useBias: false);

            // Act
            var parameters = linear.Parameters;

            // Assert
            Assert.Single(parameters);
            Assert.Equal(linear.Weight, parameters.First());
        }

        [Fact]
        public void Parameters_ReturnsWeightAndBias()
        {
            // Arrange
            var linear = new Linear(inFeatures: 64, outFeatures: 128, useBias: true);

            // Act
            var parameters = linear.Parameters;

            // Assert
            Assert.Equal(2, parameters.Count());
        }

        [Fact]
        public void SetRequiresGrad_UpdatesAllParameters()
        {
            // Arrange
            var linear = new Linear(inFeatures: 64, outFeatures: 128, useBias: true);

            // Act
            linear.SetRequiresGrad(true);

            // Assert
            Assert.True(linear.Weight.RequiresGrad);
            Assert.True(linear.Bias!.RequiresGrad);
        }

        [Fact]
        public void SetRequiresGradFalse_UpdatesAllParameters()
        {
            // Arrange
            var linear = new Linear(inFeatures: 64, outFeatures: 128, useBias: true);

            // Act
            linear.SetRequiresGrad(false);

            // Assert
            Assert.False(linear.Weight.RequiresGrad);
            Assert.False(linear.Bias!.RequiresGrad);
        }

        [Fact]
        public void ApplyToParameters_CallsActionForWeight()
        {
            // Arrange
            var linear = new Linear(inFeatures: 64, outFeatures: 128, useBias: false);
            var called = false;
            Tensor? passedTensor = null;

            // Act
            linear.ApplyToParameters((tensor) =>
            {
                called = true;
                passedTensor = tensor;
            });

            // Assert
            Assert.True(called);
            Assert.Equal(linear.Weight, passedTensor);
        }

        [Fact]
        public void ApplyToParameters_CallsActionForWeightAndBias()
        {
            // Arrange
            var linear = new Linear(inFeatures: 64, outFeatures: 128, useBias: true);
            var callCount = 0;

            // Act
            linear.ApplyToParameters((tensor) => { callCount++; });

            // Assert
            Assert.Equal(2, callCount);
        }

        [Fact]
        public void Forward_WithDifferentBatchSizes_ProducesCorrectOutputs()
        {
            // Arrange
            var linear = new Linear(inFeatures: 64, outFeatures: 128);

            // Act
            var output1 = linear.Forward(new Tensor(new float[1 * 64], new[] { 1, 64 }));
            var output2 = linear.Forward(new Tensor(new float[10 * 64], new[] { 10, 64 }));
            var output3 = linear.Forward(new Tensor(new float[100 * 64], new[] { 100, 64 }));

            // Assert
            Assert.Equal(new[] { 1, 128 }, output1.Shape);
            Assert.Equal(new[] { 10, 128 }, output2.Shape);
            Assert.Equal(new[] { 100, 128 }, output3.Shape);
        }

        [Theory]
        [InlineData(16, 32)]
        [InlineData(32, 64)]
        [InlineData(128, 256)]
        [InlineData(512, 1024)]
        public void Constructor_WithVariousDimensions_CreatesValidLinearLayer(int inFeatures, int outFeatures)
        {
            // Act
            var linear = new Linear(inFeatures, outFeatures);

            // Assert
            Assert.Equal(inFeatures, linear.InFeatures);
            Assert.Equal(outFeatures, linear.OutFeatures);
            Assert.Equal(new[] { outFeatures, inFeatures }, linear.Weight.Shape);
        }

        [Fact]
        public void Constructor_WithExistingWeightAndBias_CreatesLinearLayer()
        {
            // Arrange
            var weight = new Tensor(new float[128 * 64], new[] { 128, 64 });
            var bias = new Tensor(new float[128], new[] { 128 });

            // Act
            var linear = new Linear(weight, bias);

            // Assert
            Assert.Equal(64, linear.InFeatures);
            Assert.Equal(128, linear.OutFeatures);
            Assert.Equal(weight, linear.Weight);
            Assert.Equal(bias, linear.Bias);
        }

        [Fact]
        public void Constructor_WithNullWeight_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new Linear(null!));
        }

        [Fact]
        public void Constructor_WithInvalidWeightShape_ThrowsArgumentException()
        {
            // Arrange
            var weight = new Tensor(new float[128 * 64 * 32], new[] { 128, 64, 32 }); // 3D weight

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new Linear(weight));
        }

        [Fact]
        public void ModuleType_IsCorrect()
        {
            // Arrange
            var linear = new Linear(64, 128);

            // Act & Assert
            Assert.Equal("Linear", linear.ModuleType);
        }

        [Fact]
        public void IsTraining_CanBeSet()
        {
            // Arrange
            var linear = new Linear(64, 128);

            // Act
            linear.IsTraining = true;

            // Assert
            Assert.True(linear.IsTraining);

            // Act
            linear.IsTraining = false;

            // Assert
            Assert.False(linear.IsTraining);
        }

        [Fact]
        public void Weight_IsInitializedWithCorrectShape()
        {
            // Arrange
            var linear = new Linear(inFeatures: 32, outFeatures: 64);

            // Assert
            Assert.NotNull(linear.Weight);
            Assert.Equal(new[] { 64, 32 }, linear.Weight.Shape);
            Assert.True(linear.Weight.Size > 0);
        }

        [Fact]
        public void Bias_IsInitializedWithCorrectShape()
        {
            // Arrange
            var linear = new Linear(inFeatures: 32, outFeatures: 64, useBias: true);

            // Assert
            Assert.NotNull(linear.Bias);
            Assert.Equal(new[] { 64 }, linear.Bias.Shape);
            Assert.True(linear.Bias.Size > 0);
        }

        [Fact]
        public void Bias_WhenUseBiasFalse_IsNull()
        {
            // Arrange
            var linear = new Linear(inFeatures: 32, outFeatures: 64, useBias: false);

            // Assert
            Assert.Null(linear.Bias);
        }
    }
}
