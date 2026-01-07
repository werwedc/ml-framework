using Xunit;
using MLFramework.Functional.Compilation;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Functional
{
    public class JITTests
    {
        [Fact]
        public void Compile_ShouldReturnCompiledDelegate()
        {
            // Arrange
            Func<Tensor, Tensor> addOne = t => t + Tensor.Ones(t.Shape);

            // Act
            var compiled = Functional.Compile(addOne);

            // Assert
            Assert.NotNull(compiled);
        }

        [Fact]
        public void CompiledFunction_ShouldExecuteCorrectly()
        {
            // Arrange
            Func<Tensor, Tensor> multiplyByTwo = t => t * 2f;
            var compiled = Functional.Compile(multiplyByTwo);

            // Act
            var input = Tensor.FromArray(new[] { 1f, 2f, 3f });
            var result = compiled(input);

            // Assert
            Assert.Equal(new[] { 3 }, result.Shape);
            Assert.Equal(2f, result[0].ToScalar());
            Assert.Equal(4f, result[1].ToScalar());
            Assert.Equal(6f, result[2].ToScalar());
        }

        [Fact]
        public void Compile_DoubleInput_ShouldWork()
        {
            // Arrange
            Func<Tensor, Tensor, Tensor> add = (a, b) => a + b;
            var compiled = Functional.Compile(add);

            // Act
            var input1 = Tensor.FromArray(new[] { 1f, 2f });
            var input2 = Tensor.FromArray(new[] { 10f, 20f });
            var result = compiled(input1, input2);

            // Assert
            Assert.Equal(11f, result[0].ToScalar());
            Assert.Equal(22f, result[1].ToScalar());
        }

        [Fact]
        public void Compile_ShouldCacheCompiledFunctions()
        {
            // Arrange
            Func<Tensor, Tensor> identity = t => t;

            // Act
            var compiled1 = Functional.Compile(identity);
            var compiled2 = Functional.Compile(identity);

            // Assert
            // Should return same compiled function from cache
            // Note: Since delegates are different instances, we need to check cache size
            Assert.Equal(1, JITTransform.CacheSize);
        }

        [Fact]
        public void ClearJITCache_ShouldRemoveAllCachedFunctions()
        {
            // Arrange
            Func<Tensor, Tensor> identity = t => t;
            Functional.Compile(identity);

            // Act
            Functional.ClearJITCache();

            // Assert
            Assert.Equal(0, JITTransform.CacheSize);
        }

        [Fact]
        public void Compile_DifferentFunctions_ShouldHaveDifferentCacheEntries()
        {
            // Arrange
            Func<Tensor, Tensor> func1 = t => t + 1f;
            Func<Tensor, Tensor> func2 = t => t * 2f;

            // Act
            Functional.Compile(func1);
            Functional.Compile(func2);

            // Assert
            Assert.Equal(2, JITTransform.CacheSize);
        }

        [Fact]
        public void Compile_ShouldThrowForNonTensorReturn()
        {
            // Arrange
            Func<Tensor, int> getCount = t => t.Shape.TotalElements;

            // Act & Assert
            Assert.Throws<NotSupportedException>(() =>
                Functional.Compile(getCount));
        }

        [Fact]
        public void Compile_ShouldThrowForUnsupportedSignature()
        {
            // Arrange
            Func<Tensor, Tensor, Tensor, Tensor> tripleAdd = (a, b, c) => a + b + c;

            // Act & Assert
            Assert.Throws<NotSupportedException>(() =>
                Functional.Compile(tripleAdd));
        }

        [Fact]
        public void Compile_NeuralNetworkLayer()
        {
            // Arrange
            var weights = Tensor.FromArray(new float[12]).Reshape(4, 3);

            Func<Tensor, Tensor> denseLayer = input =>
            {
                var linear = input.MatMul(weights);
                return linear.ReLU();
            };

            // Act
            var compiled = Functional.Compile(denseLayer);

            var input = Tensor.FromArray(new float[12]).Reshape(4, 4);
            var result = compiled(input);

            // Assert
            Assert.Equal(new[] { 4, 3 }, result.Shape);
        }

        [Fact]
        public void Compile_MultipleCallsShouldUseCache()
        {
            // Arrange
            Func<Tensor, Tensor> identity = t => t;
            int cacheSizeBefore = JITTransform.CacheSize;

            // Act
            var compiled1 = Functional.Compile(identity);
            int cacheSizeAfterFirst = JITTransform.CacheSize;
            var compiled2 = Functional.Compile(identity);
            int cacheSizeAfterSecond = JITTransform.CacheSize;

            // Assert
            Assert.Equal(cacheSizeBefore + 1, cacheSizeAfterFirst);
            Assert.Equal(cacheSizeAfterFirst, cacheSizeAfterSecond);  // No increase
        }

        [Fact]
        public void CompiledFunction_ShouldPreserveSemantics()
        {
            // Arrange
            Func<Tensor, Tensor> original = t =>
            {
                var x = t.Multiply(t);
                var y = x.Add(t);
                return y.Multiply(2f);
            };

            var compiled = Functional.Compile(original);

            var input = Tensor.FromArray(new[] { 1f, 2f, 3f });

            // Act
            var originalResult = original(input);
            var compiledResult = compiled(input);

            // Assert
            Assert.Equal(originalResult.Shape, compiledResult.Shape);
            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(originalResult[i].ToScalar(),
                            compiledResult[i].ToScalar());
            }
        }
    }
}
