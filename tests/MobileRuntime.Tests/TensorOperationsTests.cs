using Microsoft.VisualStudio.TestTools.UnitTesting;
using FluentAssertions;

namespace MobileRuntime.Tests
{
    [TestClass]
    public class TensorOperationsTests
    {
        [TestMethod]
        public void Tensor_CreateTensor_WithValidData_ReturnsTensor()
        {
            // Arrange
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var shape = new[] { 2, 2 };

            // Act
            var tensor = new Tensor();
            tensor.Data = data;
            tensor.Shape = shape;
            tensor.DataType = DataType.Float32;

            // Assert
            tensor.Should().NotBeNull();
            tensor.Shape.Should().BeEquivalentTo(shape);
            tensor.Length.Should().Be(4);
        }

        [TestMethod]
        public void Tensor_DataType_ShouldMatch()
        {
            // Arrange
            var tensor = new Tensor();

            // Act
            tensor.DataType = DataType.Float32;

            // Assert
            tensor.DataType.Should().Be(DataType.Float32);
        }
    }
}
