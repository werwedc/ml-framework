using Microsoft.VisualStudio.TestTools.UnitTesting;
using FluentAssertions;
using System;

namespace MobileRuntime.Tests
{
    [TestClass]
    public class TensorOperationsTests
    {
        [TestMethod]
        public void Add_TwoTensors_ReturnsCorrectResult()
        {
            // Arrange
            var a = Tensor.FromArray(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
            var b = Tensor.FromArray(new[] { 5f, 6f, 7f, 8f }, new[] { 4 });

            // Act
            var result = TensorOperations.Add(a, b);

            // Assert
            result.ToArray<float>().Should().BeEquivalentTo(new[] { 6f, 8f, 10f, 12f });

            // Cleanup
            a.Dispose();
            b.Dispose();
            result.Dispose();
        }

        [TestMethod]
        public void Relu_PositiveValues_ReturnsUnchanged()
        {
            // Arrange
            var input = Tensor.FromArray(new[] { 1f, 2f, 3f }, new[] { 3 });

            // Act
            var result = TensorOperations.Relu(input);

            // Assert
            result.ToArray<float>().Should().BeEquivalentTo(new[] { 1f, 2f, 3f });

            // Cleanup
            input.Dispose();
            result.Dispose();
        }

        [TestMethod]
        public void Relu_NegativeValues_ReturnsZero()
        {
            // Arrange
            var input = Tensor.FromArray(new[] { -1f, -2f, -3f }, new[] { 3 });

            // Act
            var result = TensorOperations.Relu(input);

            // Assert
            result.ToArray<float>().Should().BeEquivalentTo(new[] { 0f, 0f, 0f });

            // Cleanup
            input.Dispose();
            result.Dispose();
        }

        [TestMethod]
        public void MatMul_2x2Matrices_ReturnsCorrectResult()
        {
            // Arrange
            var a = Tensor.FromArray(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
            var b = Tensor.FromArray(new[] { 5f, 6f, 7f, 8f }, new[] { 2, 2 });

            // Act
            var result = TensorOperations.MatMul(a, b);

            // Assert
            result.ToArray<float>().Should().BeEquivalentTo(new[] { 19f, 22f, 43f, 50f });

            // Cleanup
            a.Dispose();
            b.Dispose();
            result.Dispose();
        }

        [TestMethod]
        public void Multiply_TwoTensors_ReturnsCorrectResult()
        {
            // Arrange
            var a = Tensor.FromArray(new[] { 2f, 3f, 4f }, new[] { 3 });
            var b = Tensor.FromArray(new[] { 5f, 6f, 7f }, new[] { 3 });

            // Act
            var result = TensorOperations.Multiply(a, b);

            // Assert
            result.ToArray<float>().Should().BeEquivalentTo(new[] { 10f, 18f, 28f });

            // Cleanup
            a.Dispose();
            b.Dispose();
            result.Dispose();
        }

        [TestMethod]
        public void Subtract_TwoTensors_ReturnsCorrectResult()
        {
            // Arrange
            var a = Tensor.FromArray(new[] { 10f, 20f, 30f }, new[] { 3 });
            var b = Tensor.FromArray(new[] { 1f, 2f, 3f }, new[] { 3 });

            // Act
            var result = TensorOperations.Subtract(a, b);

            // Assert
            result.ToArray<float>().Should().BeEquivalentTo(new[] { 9f, 18f, 27f });

            // Cleanup
            a.Dispose();
            b.Dispose();
            result.Dispose();
        }

        [TestMethod]
        public void Divide_TwoTensors_ReturnsCorrectResult()
        {
            // Arrange
            var a = Tensor.FromArray(new[] { 10f, 20f, 30f }, new[] { 3 });
            var b = Tensor.FromArray(new[] { 2f, 4f, 6f }, new[] { 3 });

            // Act
            var result = TensorOperations.Divide(a, b);

            // Assert
            result.ToArray<float>().Should().BeEquivalentTo(new[] { 5f, 5f, 5f }, options =>
                options.AllowingRelativeDifference(0.0001f));

            // Cleanup
            a.Dispose();
            b.Dispose();
            result.Dispose();
        }

        [TestMethod]
        public void Abs_NegativeValues_ReturnsPositive()
        {
            // Arrange
            var input = Tensor.FromArray(new[] { -1f, -2f, -3f }, new[] { 3 });

            // Act
            var result = TensorOperations.Abs(input);

            // Assert
            result.ToArray<float>().Should().BeEquivalentTo(new[] { 1f, 2f, 3f });

            // Cleanup
            input.Dispose();
            result.Dispose();
        }

        [TestMethod]
        public void Sqrt_PositiveValues_ReturnsSquareRoot()
        {
            // Arrange
            var input = Tensor.FromArray(new[] { 1f, 4f, 9f }, new[] { 3 });

            // Act
            var result = TensorOperations.Sqrt(input);

            // Assert
            result.ToArray<float>().Should().BeEquivalentTo(new[] { 1f, 2f, 3f }, options =>
                options.AllowingRelativeDifference(0.0001f));

            // Cleanup
            input.Dispose();
            result.Dispose();
        }

        [TestMethod]
        public void Square_Values_ReturnsSquaredValues()
        {
            // Arrange
            var input = Tensor.FromArray(new[] { 2f, 3f, 4f }, new[] { 3 });

            // Act
            var result = TensorOperations.Square(input);

            // Assert
            result.ToArray<float>().Should().BeEquivalentTo(new[] { 4f, 9f, 16f }, options =>
                options.AllowingRelativeDifference(0.0001f));

            // Cleanup
            input.Dispose();
            result.Dispose();
        }

        [TestMethod]
        public void Sum_AllElements_ReturnsCorrectSum()
        {
            // Arrange
            var input = Tensor.FromArray(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });

            // Act
            var result = TensorOperations.Sum(input);

            // Assert
            result.ToArray<float>()[0].Should().Be(10f);

            // Cleanup
            input.Dispose();
            result.Dispose();
        }

        [TestMethod]
        public void Mean_AllElements_ReturnsCorrectMean()
        {
            // Arrange
            var input = Tensor.FromArray(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });

            // Act
            var result = TensorOperations.Mean(input);

            // Assert
            result.ToArray<float>()[0].Should().Be(2.5f);

            // Cleanup
            input.Dispose();
            result.Dispose();
        }

        [TestMethod]
        public void Max_AllElements_ReturnsCorrectMax()
        {
            // Arrange
            var input = Tensor.FromArray(new[] { 1f, 5f, 3f, 2f }, new[] { 4 });

            // Act
            var result = TensorOperations.Max(input);

            // Assert
            result.ToArray<float>()[0].Should().Be(5f);

            // Cleanup
            input.Dispose();
            result.Dispose();
        }

        [TestMethod]
        public void Transpose_2DMatrix_ReturnsTransposedMatrix()
        {
            // Arrange
            var input = Tensor.FromArray(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });

            // Act
            var result = TensorOperations.Transpose(input);

            // Assert
            result.ToArray<float>().Should().BeEquivalentTo(new[] { 1f, 3f, 2f, 4f });
            result.Shape.Should().BeEquivalentTo(new[] { 2, 2 });

            // Cleanup
            input.Dispose();
            result.Dispose();
        }

        [TestMethod]
        public void Reshape_ChangeShape_MaintainsData()
        {
            // Arrange
            var input = Tensor.FromArray(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 2, 3 });

            // Act
            var result = TensorOperations.Reshape(input, new[] { 3, 2 });

            // Assert
            result.Shape.Should().BeEquivalentTo(new[] { 3, 2 });
            result.ToArray<float>().Should().BeEquivalentTo(new[] { 1f, 2f, 3f, 4f, 5f, 6f });

            // Cleanup
            input.Dispose();
            result.Dispose();
        }

        [TestMethod]
        public void Copy_Tensor_CreatesIndependentCopy()
        {
            // Arrange
            var input = Tensor.FromArray(new[] { 1f, 2f, 3f }, new[] { 3 });

            // Act
            var copy = TensorOperations.Copy(input);

            // Assert
            copy.ToArray<float>().Should().BeEquivalentTo(new[] { 1f, 2f, 3f });
            copy.DataPointer.Should().NotBe(input.DataPointer);

            // Cleanup
            input.Dispose();
            copy.Dispose();
        }

        [TestMethod]
        public void Tensor_FromArray_CreatesCorrectTensor()
        {
            // Arrange
            var data = new[] { 1f, 2f, 3f, 4f };
            var shape = new[] { 2, 2 };

            // Act
            var tensor = Tensor.FromArray(data, shape);

            // Assert
            tensor.Should().NotBeNull();
            tensor.Shape.Should().BeEquivalentTo(shape);
            tensor.Size.Should().Be(4);
            tensor.ToArray<float>().Should().BeEquivalentTo(data);

            // Cleanup
            tensor.Dispose();
        }

        [TestMethod]
        public void Tensor_Zeros_CreatesZeroTensor()
        {
            // Arrange
            var shape = new[] { 2, 2 };

            // Act
            var tensor = Tensor.Zeros(shape, DataType.Float32);

            // Assert
            tensor.Should().NotBeNull();
            tensor.Shape.Should().BeEquivalentTo(shape);
            tensor.ToArray<float>().Should().AllBeEquivalentTo(0f);

            // Cleanup
            tensor.Dispose();
        }

        [TestMethod]
        public void Tensor_Ones_CreatesOnesTensor()
        {
            // Arrange
            var shape = new[] { 2, 2 };

            // Act
            var tensor = Tensor.Ones(shape, DataType.Float32);

            // Assert
            tensor.Should().NotBeNull();
            tensor.Shape.Should().BeEquivalentTo(shape);
            tensor.ToArray<float>().Should().AllBeEquivalentTo(1f);

            // Cleanup
            tensor.Dispose();
        }

        [TestMethod]
        public void Tensor_AddScalar_InPlaceOperation_WorksCorrectly()
        {
            // Arrange
            var tensor = Tensor.FromArray(new[] { 1f, 2f, 3f }, new[] { 3 });

            // Act
            tensor.AddScalar(10f);

            // Assert
            tensor.ToArray<float>().Should().BeEquivalentTo(new[] { 11f, 12f, 13f });

            // Cleanup
            tensor.Dispose();
        }

        [TestMethod]
        public void Tensor_MultiplyScalar_InPlaceOperation_WorksCorrectly()
        {
            // Arrange
            var tensor = Tensor.FromArray(new[] { 1f, 2f, 3f }, new[] { 3 });

            // Act
            tensor.MultiplyScalar(2f);

            // Assert
            tensor.ToArray<float>().Should().BeEquivalentTo(new[] { 2f, 4f, 6f });

            // Cleanup
            tensor.Dispose();
        }

        [TestMethod]
        public void Tensor_Clamp_InPlaceOperation_WorksCorrectly()
        {
            // Arrange
            var tensor = Tensor.FromArray(new[] { -1f, 0f, 1f, 2f, 3f }, new[] { 5 });

            // Act
            tensor.Clamp(0f, 2f);

            // Assert
            tensor.ToArray<float>().Should().BeEquivalentTo(new[] { 0f, 0f, 1f, 2f, 2f });

            // Cleanup
            tensor.Dispose();
        }

        [TestMethod]
        public void Tensor_ByteCount_ReturnsCorrectSize()
        {
            // Arrange
            var tensor = Tensor.FromArray(new[] { 1f, 2f, 3f }, new[] { 3 });

            // Act & Assert
            tensor.ByteCount.Should().Be(12); // 3 elements * 4 bytes per float

            // Cleanup
            tensor.Dispose();
        }

        [TestMethod]
        public void Tensor_DataPointer_ReturnsValidPointer()
        {
            // Arrange
            var tensor = Tensor.FromArray(new[] { 1f, 2f, 3f }, new[] { 3 });

            // Act & Assert
            tensor.DataPointer.Should().NotBe(IntPtr.Zero);

            // Cleanup
            tensor.Dispose();
        }
    }
}
