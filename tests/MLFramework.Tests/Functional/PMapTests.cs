using Xunit;
using MLFramework.Functional;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Functional
{
    public class PMapTests
    {
        [Fact]
        public void Parallelize_SingleInput_ShouldShardAndGather()
        {
            // Arrange
            var mesh = DeviceMeshFactory.Create1D(2, DeviceType.CPU);
            Func<Tensor, Tensor> multiplyByTwo = t => t * 2f;
            var parallelMultiply = Functional.Parallelize(multiplyByTwo, mesh);

            // Act
            var input = Tensor.FromArray(new[] { 1f, 2, 3, 4 }).Reshape(4, 1);
            var result = parallelMultiply(input);

            // Assert
            Assert.Equal(new[] { 4, 1 }, result.Shape);
            Assert.Equal(2f, result[0, 0].ToScalar());
            Assert.Equal(4f, result[1, 0].ToScalar());
            Assert.Equal(6f, result[2, 0].ToScalar());
            Assert.Equal(8f, result[3, 0].ToScalar());
        }

        [Fact]
        public void Parallelize_DoubleInput_ShouldShardBothInputs()
        {
            // Arrange
            var mesh = DeviceMeshFactory.Create1D(2, DeviceType.CPU);
            Func<Tensor, Tensor, Tensor> add = (a, b) => a + b;
            var parallelAdd = Functional.Parallelize(add, mesh);

            // Act
            var input1 = Tensor.FromArray(new[] { 1f, 2, 3, 4 }).Reshape(4, 1);
            var input2 = Tensor.FromArray(new[] { 10f, 20, 30, 40 }).Reshape(4, 1);
            var result = parallelAdd(input1, input2);

            // Assert
            Assert.Equal(new[] { 4, 1 }, result.Shape);
            Assert.Equal(11f, result[0, 0].ToScalar());
            Assert.Equal(22f, result[1, 0].ToScalar());
            Assert.Equal(33f, result[2, 0].ToScalar());
            Assert.Equal(44f, result[3, 0].ToScalar());
        }

        [Fact]
        public void Parallelize_WithInAxes_ShouldShardOnlyFirstInput()
        {
            // Arrange
            var mesh = DeviceMeshFactory.Create1D(2, DeviceType.CPU);
            Func<Tensor, Tensor, Tensor> multiply = (a, b) => a * b;
            var parallelMultiply = Functional.Parallelize(multiply, mesh, new object[] { 0, null });

            // Act
            var batchedInput = Tensor.FromArray(new[] { 1f, 2, 3, 4 }).Reshape(4, 1);
            var singleInput = Tensor.FromArray(new[] { 10f }).Reshape(1);
            var result = parallelMultiply(batchedInput, singleInput);

            // Assert
            Assert.Equal(new[] { 4, 1 }, result.Shape);
            // Each batch element should be multiplied by the same single input
            Assert.Equal(10f, result[0, 0].ToScalar());
            Assert.Equal(20f, result[1, 0].ToScalar());
            Assert.Equal(30f, result[2, 0].ToScalar());
            Assert.Equal(40f, result[3, 0].ToScalar());
        }

        [Fact]
        public void Parallelize_ShouldThrowForNullMesh()
        {
            // Arrange
            Func<Tensor, Tensor> identity = t => t;

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                Functional.Parallelize(identity, null));
        }

        [Fact]
        public void Parallelize_ShouldThrowForNonDivisibleShard()
        {
            // Arrange
            var mesh = DeviceMeshFactory.Create1D(3, DeviceType.CPU);  // 3 devices
            Func<Tensor, Tensor> identity = t => t;
            var parallelIdentity = Functional.Parallelize(identity, mesh);

            var input = Tensor.FromArray(new float[5]).Reshape(5, 1);  // 5 elements (not divisible by 3)

            // Act & Assert
            Assert.Throws<ArgumentException>(() => parallelIdentity(input));
        }

        [Fact]
        public void Parallelize_ShouldThrowForInvalidInAxesLength()
        {
            // Arrange
            var mesh = DeviceMeshFactory.Create1D(2, DeviceType.CPU);
            Func<Tensor, Tensor, Tensor> add = (a, b) => a + b;

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                Functional.Parallelize(add, mesh, new object[] { 0 }));  // Should have 2 axes
        }

        [Fact]
        public void Parallelize_ShouldThrowForNonTensorReturn()
        {
            // Arrange
            var mesh = DeviceMeshFactory.Create1D(2, DeviceType.CPU);
            Func<Tensor, int> getCount = t => t.Shape.TotalElements;

            // Act & Assert
            Assert.Throws<NotSupportedException>(() =>
                Functional.Parallelize(getCount, mesh));
        }

        [Fact]
        public void Parallelize_BatchTrainingStep()
        {
            // Arrange
            var mesh = DeviceMeshFactory.Create1D(2, DeviceType.CPU);
            Func<Tensor, Tensor> forwardPass = batch =>
            {
                // Simulate forward pass
                return batch.MatMul(Tensor.FromArray(new float[12]).Reshape(4, 3));
            };

            var parallelForward = Functional.Parallelize(forwardPass, mesh);

            // Act
            var input = Tensor.FromArray(new float[24]).Reshape(8, 3);  // Batch of 8
            var result = parallelForward(input);

            // Assert
            Assert.Equal(new[] { 8, 3 }, result.Shape);
        }

        [Fact]
        public void Parallelize_WithWeights_ShouldNotShardWeights()
        {
            // Arrange
            var mesh = DeviceMeshFactory.Create1D(2, DeviceType.CPU);
            var weights = Tensor.FromArray(new float[12]).Reshape(4, 3);

            Func<Tensor, Tensor, Tensor> applyWeights = (input, w) => input.MatMul(w);
            var parallelApply = Functional.Parallelize(applyWeights, mesh, new object[] { 0, null });

            // Act
            var input = Tensor.FromArray(new float[24]).Reshape(8, 4);
            var result = parallelApply(input, weights);

            // Assert
            Assert.Equal(new[] { 8, 3 }, result.Shape);
        }
    }
}
