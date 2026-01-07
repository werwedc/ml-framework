using Xunit;
using MLFramework.Functional;
using RitterFramework.Core.Tensor;
using System;
using System.Linq;

namespace MLFramework.Tests.Functional
{
    public class VMapTests
    {
        #region Single Input Tests

        [Fact]
        public void Vectorize_SingleInput_1DTensor_ShouldApplyFunction()
        {
            // Arrange
            Func<Tensor, Tensor> square = t =>
            {
                var newData = t.Data.Select(x => x * x).ToArray();
                return new Tensor(newData, t.Shape);
            };

            var batch = Tensor.FromArray(new[] { 1f, 2f, 3f, 4f, 5f });

            // Act
            var vectorized = Functional.Vectorize(square, axis: 0);
            var result = vectorized(batch);

            // Assert - Should square each element
            Assert.Equal(new[] { 5 }, result.Shape);
            Assert.Equal(1f, result.Data[0]);   // 1^2
            Assert.Equal(4f, result.Data[1]);   // 2^2
            Assert.Equal(9f, result.Data[2]);   // 3^2
            Assert.Equal(16f, result.Data[3]);  // 4^2
            Assert.Equal(25f, result.Data[4]);  // 5^2
        }

        [Fact]
        public void Vectorize_SingleInput_2DTensor_ShouldApplyAlongAxis0()
        {
            // Arrange
            Func<Tensor, Tensor> sum = t =>
            {
                return new Tensor(new[] { t.Data.Sum() }, new[] { 1 });
            };

            var batch = new Tensor(new float[6] { 1, 2, 3, 4, 5, 6 }, new[] { 3, 2 });

            // Act
            var vectorized = Functional.Vectorize(sum, axis: 0);
            var result = vectorized(batch);

            // Assert - Should sum each row and stack
            Assert.Equal(new[] { 3, 1 }, result.Shape);  // 3 rows stacked with 1 element each
            Assert.Equal(3f, result.Data[0]);  // 1 + 2
            Assert.Equal(7f, result.Data[1]);  // 3 + 4
            Assert.Equal(11f, result.Data[2]); // 5 + 6
        }

        [Fact]
        public void Vectorize_SingleInput_MultiplyByScalar_ShouldWork()
        {
            // Arrange
            Func<Tensor, Tensor> multiplyBy2 = t => t * 2f;

            var batch = Tensor.FromArray(new[] { 1f, 2f, 3f });

            // Act
            var vectorized = Functional.Vectorize(multiplyBy2, axis: 0);
            var result = vectorized(batch);

            // Assert
            Assert.Equal(new[] { 3 }, result.Shape);
            Assert.Equal(2f, result.Data[0]);
            Assert.Equal(4f, result.Data[1]);
            Assert.Equal(6f, result.Data[2]);
        }

        [Fact]
        public void Vectorize_SingleInput_WithReshape_ShouldWork()
        {
            // Arrange
            Func<Tensor, Tensor> reshapeTo2x1 = t => t.Reshape(new[] { 2, 1 });

            var batch = new Tensor(new float[6] { 1, 2, 3, 4, 5, 6 }, new[] { 3, 2 });

            // Act
            var vectorized = Functional.Vectorize(reshapeTo2x1, axis: 0);
            var result = vectorized(batch);

            // Assert
            Assert.Equal(new[] { 3, 2, 1 }, result.Shape);  // 3 rows, each 2x1
        }

        #endregion

        #region Double Input Tests

        [Fact]
        public void Vectorize_DoubleInput_Addition_ShouldWork()
        {
            // Arrange
            Func<Tensor, Tensor, Tensor> add = (a, b) => a + b;

            var batch1 = Tensor.FromArray(new[] { 1f, 2f, 3f });
            var batch2 = Tensor.FromArray(new[] { 10f, 20f, 30f });

            // Act
            var vectorized = Functional.Vectorize(add, axis: 0);
            var result = vectorized(batch1, batch2);

            // Assert
            Assert.Equal(new[] { 3 }, result.Shape);
            Assert.Equal(11f, result.Data[0]);  // 1 + 10
            Assert.Equal(22f, result.Data[1]);  // 2 + 20
            Assert.Equal(33f, result.Data[2]);  // 3 + 30
        }

        [Fact]
        public void Vectorize_DoubleInput_Multiplication_ShouldWork()
        {
            // Arrange
            Func<Tensor, Tensor, Tensor> multiply = (a, b) =>
            {
                var newData = a.Data.Zip(b.Data, (x, y) => x * y).ToArray();
                return new Tensor(newData, a.Shape);
            };

            var batch1 = Tensor.FromArray(new[] { 1f, 2f, 3f });
            var batch2 = Tensor.FromArray(new[] { 10f, 20f, 30f });

            // Act
            var vectorized = Functional.Vectorize(multiply, axis: 0);
            var result = vectorized(batch1, batch2);

            // Assert
            Assert.Equal(new[] { 3 }, result.Shape);
            Assert.Equal(10f, result.Data[0]);  // 1 * 10
            Assert.Equal(40f, result.Data[1]);  // 2 * 20
            Assert.Equal(90f, result.Data[2]);  // 3 * 30
        }

        [Fact]
        public void Vectorize_DoubleInput_2DTensors_ShouldWork()
        {
            // Arrange
            Func<Tensor, Tensor, Tensor> add = (a, b) => a + b;

            var batch1 = new Tensor(new float[6] { 1, 2, 3, 4, 5, 6 }, new[] { 3, 2 });
            var batch2 = new Tensor(new float[6] { 10, 20, 30, 40, 50, 60 }, new[] { 3, 2 });

            // Act
            var vectorized = Functional.Vectorize(add, axis: 0);
            var result = vectorized(batch1, batch2);

            // Assert
            Assert.Equal(new[] { 3, 2 }, result.Shape);
            Assert.Equal(11f, result.Data[0]);   // 1 + 10
            Assert.Equal(22f, result.Data[1]);   // 2 + 20
            Assert.Equal(33f, result.Data[2]);   // 3 + 30
            Assert.Equal(44f, result.Data[3]);   // 4 + 40
            Assert.Equal(55f, result.Data[4]);   // 5 + 50
            Assert.Equal(66f, result.Data[5]);   // 6 + 60
        }

        #endregion

        #region Error Handling Tests

        [Fact]
        public void Vectorize_SingleInput_InsufficientDimensions_ShouldThrow()
        {
            // Arrange
            Func<Tensor, Tensor> identity = t => t;
            var tensor = Tensor.FromArray(new[] { 1f, 2f, 3f });  // 1D tensor

            // Act & Assert
            var vectorized = Functional.Vectorize(identity, axis: 1);  // axis 1 but tensor is 1D
            Assert.Throws<ArgumentException>(() => vectorized(tensor));
        }

        [Fact]
        public void Vectorize_DoubleInput_DifferentBatchSizes_ShouldThrow()
        {
            // Arrange
            Func<Tensor, Tensor, Tensor> add = (a, b) => a + b;

            var batch1 = Tensor.FromArray(new[] { 1f, 2f, 3f });  // 3 elements
            var batch2 = Tensor.FromArray(new[] { 10f, 20f });     // 2 elements

            // Act & Assert
            var vectorized = Functional.Vectorize(add, axis: 0);
            Assert.Throws<ArgumentException>(() => vectorized(batch1, batch2));
        }

        [Fact]
        public void Vectorize_DoubleInput_FirstInputInsufficientDimensions_ShouldThrow()
        {
            // Arrange
            Func<Tensor, Tensor, Tensor> add = (a, b) => a + b;

            var batch1 = Tensor.FromArray(new[] { 1f, 2f, 3f });  // 1D tensor
            var batch2 = new Tensor(new float[6] { 1, 2, 3, 4, 5, 6 }, new[] { 3, 2 });  // 2D tensor

            // Act & Assert
            var vectorized = Functional.Vectorize(add, axis: 1);
            Assert.Throws<ArgumentException>(() => vectorized(batch1, batch2));
        }

        [Fact]
        public void Vectorize_DoubleInput_SecondInputInsufficientDimensions_ShouldThrow()
        {
            // Arrange
            Func<Tensor, Tensor, Tensor> add = (a, b) => a + b;

            var batch1 = new Tensor(new float[6] { 1, 2, 3, 4, 5, 6 }, new[] { 3, 2 });  // 2D tensor
            var batch2 = Tensor.FromArray(new[] { 1f, 2f, 3f });  // 1D tensor

            // Act & Assert
            var vectorized = Functional.Vectorize(add, axis: 1);
            Assert.Throws<ArgumentException>(() => vectorized(batch1, batch2));
        }

        [Fact]
        public void Vectorize_NonTensorReturn_ShouldThrow()
        {
            // Arrange
            Func<Tensor, int> getSize = t => t.Size;

            // Act & Assert
            Assert.Throws<NotSupportedException>(() =>
                Functional.Vectorize(getSize, axis: 0));
        }

        [Fact]
        public void Vectorize_UnsupportedDelegateSignature_ShouldThrow()
        {
            // Arrange
            Func<Tensor, Tensor, Tensor, Tensor> addThree = (a, b, c) => a + b + c;

            // Act & Assert
            Assert.Throws<NotSupportedException>(() =>
                Functional.Vectorize(addThree, axis: 0));
        }

        #endregion

        #region Tensor Extension Methods Tests

        [Fact]
        public void Take_1DTensor_ShouldExtractScalar()
        {
            // Arrange
            var tensor = Tensor.FromArray(new[] { 1f, 2f, 3f, 4f, 5f });

            // Act
            var result = tensor.Take(axis: 0, index: 2);

            // Assert
            Assert.Equal(new[] { }, result.Shape);  // Scalar
            Assert.Single(result.Data);
            Assert.Equal(3f, result.Data[0]);
        }

        [Fact]
        public void Take_2DTensor_Axis0_ShouldExtractRow()
        {
            // Arrange
            var tensor = new Tensor(new float[6] { 1, 2, 3, 4, 5, 6 }, new[] { 3, 2 });

            // Act
            var result = tensor.Take(axis: 0, index: 1);

            // Assert
            Assert.Equal(new[] { 2 }, result.Shape);  // 1D with 2 elements
            Assert.Equal(2, result.Data.Length);
            Assert.Equal(3f, result.Data[0]);
            Assert.Equal(4f, result.Data[1]);
        }

        [Fact]
        public void Take_2DTensor_Axis1_ShouldExtractColumn()
        {
            // Arrange
            var tensor = new Tensor(new float[6] { 1, 2, 3, 4, 5, 6 }, new[] { 3, 2 });

            // Act
            var result = tensor.Take(axis: 1, index: 0);

            // Assert
            Assert.Equal(new[] { 3 }, result.Shape);  // 1D with 3 elements
            Assert.Equal(3, result.Data.Length);
            Assert.Equal(1f, result.Data[0]);
            Assert.Equal(3f, result.Data[1]);
            Assert.Equal(5f, result.Data[2]);
        }

        [Fact]
        public void Take_InvalidAxis_ShouldThrow()
        {
            // Arrange
            var tensor = Tensor.FromArray(new[] { 1f, 2f, 3f });

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                tensor.Take(axis: 5, index: 0));
        }

        [Fact]
        public void Take_InvalidIndex_ShouldThrow()
        {
            // Arrange
            var tensor = Tensor.FromArray(new[] { 1f, 2f, 3f });

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                tensor.Take(axis: 0, index: 10));
        }

        [Fact]
        public void Stack_EmptyCollection_ShouldThrow()
        {
            // Arrange
            var tensors = Array.Empty<Tensor>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                tensors.Stack(axis: 0));
        }

        [Fact]
        public void Stack_SingleTensor_ShouldReturnClone()
        {
            // Arrange
            var tensor = Tensor.FromArray(new[] { 1f, 2f, 3f });

            // Act
            var result = new[] { tensor }.Stack(axis: 0);

            // Assert
            Assert.Equal(tensor.Shape, result.Shape);
            Assert.Equal(tensor.Data, result.Data);
        }

        [Fact]
        public void Stack_MultipleTensors_Axis0_ShouldCreateNewDimension()
        {
            // Arrange
            var tensor1 = Tensor.FromArray(new[] { 1f, 2f });
            var tensor2 = Tensor.FromArray(new[] { 3f, 4f });
            var tensor3 = Tensor.FromArray(new[] { 5f, 6f });

            // Act
            var result = new[] { tensor1, tensor2, tensor3 }.Stack(axis: 0);

            // Assert
            Assert.Equal(new[] { 3, 2 }, result.Shape);
            Assert.Equal(1f, result.Data[0]);
            Assert.Equal(2f, result.Data[1]);
            Assert.Equal(3f, result.Data[2]);
            Assert.Equal(4f, result.Data[3]);
            Assert.Equal(5f, result.Data[4]);
            Assert.Equal(6f, result.Data[5]);
        }

        [Fact]
        public void Stack_MultipleTensors_Axis1_ShouldInsertDimension()
        {
            // Arrange
            var tensor1 = Tensor.FromArray(new[] { 1f, 2f });
            var tensor2 = Tensor.FromArray(new[] { 3f, 4f });

            // Act
            var result = new[] { tensor1, tensor2 }.Stack(axis: 1);

            // Assert
            Assert.Equal(new[] { 2, 2 }, result.Shape);
        }

        [Fact]
        public void Stack_IncompatibleShapes_ShouldThrow()
        {
            // Arrange
            var tensor1 = Tensor.FromArray(new[] { 1f, 2f, 3f });
            var tensor2 = Tensor.FromArray(new[] { 1f, 2f });

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new[] { tensor1, tensor2 }.Stack(axis: 0));
        }

        [Fact]
        public void Stack_NegativeAxis_ShouldWork()
        {
            // Arrange
            var tensor1 = Tensor.FromArray(new[] { 1f, 2f });
            var tensor2 = Tensor.FromArray(new[] { 3f, 4f });

            // Act - axis -1 means last position
            var result = new[] { tensor1, tensor2 }.Stack(axis: -1);

            // Assert
            Assert.Equal(new[] { 2, 2 }, result.Shape);
        }

        [Fact]
        public void Stack_InvalidAxis_ShouldThrow()
        {
            // Arrange
            var tensor1 = Tensor.FromArray(new[] { 1f, 2f });
            var tensor2 = Tensor.FromArray(new[] { 3f, 4f });

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new[] { tensor1, tensor2 }.Stack(axis: 10));
        }

        #endregion

        #region Real-World Scenario Tests

        [Fact]
        public void Vectorize_BatchNormalization_ShouldWork()
        {
            // Arrange
            Func<Tensor, Tensor> normalize = t =>
            {
                var mean = t.Data.Sum() / t.Data.Length;
                var variance = t.Data.Sum(x => MathF.Pow(x - mean, 2)) / t.Data.Length;
                var std = MathF.Sqrt(variance) + 1e-8f;
                var newData = t.Data.Select(x => (x - mean) / std).ToArray();
                return new Tensor(newData, t.Shape);
            };

            // Create a batch of vectors
            var batch = new Tensor(new float[6] { 1, 2, 3, 4, 5, 6 }, new[] { 3, 2 });

            // Act
            var vectorized = Functional.Vectorize(normalize, axis: 0);
            var result = vectorized(batch);

            // Assert
            Assert.Equal(new[] { 3, 2 }, result.Shape);
        }

        [Fact]
        public void Vectorize_Compose_ShouldChainOperations()
        {
            // Arrange
            Func<Tensor, Tensor> multiplyBy2 = t => t * 2f;
            Func<Tensor, Tensor> add1 = t => t + 1f;

            var batch = Tensor.FromArray(new[] { 1f, 2f, 3f });

            // Act - Compose operations and vectorize
            var composed = Functional.Compose(add1, multiplyBy2);
            var vectorized = Functional.Vectorize(composed, axis: 0);
            var result = vectorized(batch);

            // Assert - Each element: (x * 2) + 1
            Assert.Equal(new[] { 3 }, result.Shape);
            Assert.Equal(3f, result.Data[0]);  // (1 * 2) + 1
            Assert.Equal(5f, result.Data[1]);  // (2 * 2) + 1
            Assert.Equal(7f, result.Data[2]);  // (3 * 2) + 1
        }

        [Fact]
        public void Vectorize_PartialApplication_ShouldWork()
        {
            // Arrange
            Func<Tensor, Tensor, Tensor> weightedSum = (a, b) =>
            {
                var newData = a.Data.Zip(b.Data, (x, y) => x + 2f * y).ToArray();
                return new Tensor(newData, a.Shape);
            };

            var weights = Tensor.FromArray(new[] { 0.5f, 0.5f });

            // Act
            var weightedWithWeights = Functional.Partial(weightedSum, weights);
            var vectorized = Functional.Vectorize(weightedWithWeights, axis: 0);

            var batch = new Tensor(new float[4] { 1, 2, 3, 4 }, new[] { 2, 2 });
            var result = vectorized(batch, weights);

            // Assert
            Assert.Equal(new[] { 2, 2 }, result.Shape);
        }

        #endregion
    }
}
