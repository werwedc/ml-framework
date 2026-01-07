using System;
using Xunit;
using MLFramework.Quantization.DataStructures;

namespace MLFramework.Tests.Quantization.DataStructures
{
    public class CalibrationMethodTests
    {
        [Fact]
        public void AllMethodValues_AreValidEnumValues()
        {
            // Arrange & Act & Assert
            Assert.True(Enum.IsDefined(typeof(CalibrationMethod), CalibrationMethod.MinMax));
            Assert.True(Enum.IsDefined(typeof(CalibrationMethod), CalibrationMethod.Entropy));
            Assert.True(Enum.IsDefined(typeof(CalibrationMethod), CalibrationMethod.Percentile));
            Assert.True(Enum.IsDefined(typeof(CalibrationMethod), CalibrationMethod.MovingAverage));
        }

        [Theory]
        [InlineData(CalibrationMethod.MinMax, "MinMax")]
        [InlineData(CalibrationMethod.Entropy, "Entropy")]
        [InlineData(CalibrationMethod.Percentile, "Percentile")]
        [InlineData(CalibrationMethod.MovingAverage, "MovingAverage")]
        public void ToString_ReturnsCorrectStringRepresentation(CalibrationMethod method, string expected)
        {
            // Act
            string result = method.ToString();

            // Assert
            Assert.Equal(expected, result);
        }

        [Theory]
        [InlineData("MinMax", CalibrationMethod.MinMax)]
        [InlineData("Entropy", CalibrationMethod.Entropy)]
        [InlineData("Percentile", CalibrationMethod.Percentile)]
        [InlineData("MovingAverage", CalibrationMethod.MovingAverage)]
        public void ParseString_ReturnsCorrectMethod(string value, CalibrationMethod expected)
        {
            // Act
            var result = (CalibrationMethod)Enum.Parse(typeof(CalibrationMethod), value);

            // Assert
            Assert.Equal(expected, result);
        }

        [Fact]
        public void ParseString_WithInvalidValue_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => Enum.Parse(typeof(CalibrationMethod), "InvalidMethod"));
        }

        [Theory]
        [InlineData("MINMAX")]
        [InlineData("minmax")]
        [InlineData("MinMax ")]
        public void TryParseString_IsCaseInsensitive(string value)
        {
            // Act
            bool success = Enum.TryParse(value, true, out CalibrationMethod result);

            // Assert
            Assert.True(success);
            Assert.Equal(CalibrationMethod.MinMax, result);
        }

        [Fact]
        public void EnumValues_HaveSequentialIntegers()
        {
            // Act
            int minMax = (int)CalibrationMethod.MinMax;
            int entropy = (int)CalibrationMethod.Entropy;
            int percentile = (int)CalibrationMethod.Percentile;
            int movingAverage = (int)CalibrationMethod.MovingAverage;

            // Assert
            Assert.Equal(0, minMax);
            Assert.Equal(1, entropy);
            Assert.Equal(2, percentile);
            Assert.Equal(3, movingAverage);
        }

        [Theory]
        [InlineData(CalibrationMethod.MinMax, true)]
        [InlineData(CalibrationMethod.Entropy, false)]
        [InlineData(CalibrationMethod.Percentile, false)]
        [InlineData(CalibrationMethod.MovingAverage, false)]
        public void IsMinMax_CorrectlyIdentifiesMethod(CalibrationMethod method, bool expected)
        {
            // Act
            bool isMinMax = method == CalibrationMethod.MinMax;

            // Assert
            Assert.Equal(expected, isMinMax);
        }

        [Theory]
        [InlineData(CalibrationMethod.Entropy, true)]
        [InlineData(CalibrationMethod.MinMax, false)]
        [InlineData(CalibrationMethod.Percentile, false)]
        [InlineData(CalibrationMethod.MovingAverage, false)]
        public void IsEntropy_CorrectlyIdentifiesMethod(CalibrationMethod method, bool expected)
        {
            // Act
            bool isEntropy = method == CalibrationMethod.Entropy;

            // Assert
            Assert.Equal(expected, isEntropy);
        }

        [Theory]
        [InlineData(CalibrationMethod.Percentile, true)]
        [InlineData(CalibrationMethod.MinMax, false)]
        [InlineData(CalibrationMethod.Entropy, false)]
        [InlineData(CalibrationMethod.MovingAverage, false)]
        public void IsPercentile_CorrectlyIdentifiesMethod(CalibrationMethod method, bool expected)
        {
            // Act
            bool isPercentile = method == CalibrationMethod.Percentile;

            // Assert
            Assert.Equal(expected, isPercentile);
        }

        [Theory]
        [InlineData(CalibrationMethod.MovingAverage, true)]
        [InlineData(CalibrationMethod.MinMax, false)]
        [InlineData(CalibrationMethod.Entropy, false)]
        [InlineData(CalibrationMethod.Percentile, false)]
        public void IsMovingAverage_CorrectlyIdentifiesMethod(CalibrationMethod method, bool expected)
        {
            // Act
            bool isMovingAverage = method == CalibrationMethod.MovingAverage;

            // Assert
            Assert.Equal(expected, isMovingAverage);
        }

        [Fact]
        public void GetNames_ReturnsAllMethodNames()
        {
            // Act
            string[] names = Enum.GetNames(typeof(CalibrationMethod));

            // Assert
            Assert.Equal(4, names.Length);
            Assert.Contains("MinMax", names);
            Assert.Contains("Entropy", names);
            Assert.Contains("Percentile", names);
            Assert.Contains("MovingAverage", names);
        }

        [Fact]
        public void GetValues_ReturnsAllMethodValues()
        {
            // Act
            Array values = Enum.GetValues(typeof(CalibrationMethod));

            // Assert
            Assert.Equal(4, values.Length);
            Assert.Contains(CalibrationMethod.MinMax, values);
            Assert.Contains(CalibrationMethod.Entropy, values);
            Assert.Contains(CalibrationMethod.Percentile, values);
            Assert.Contains(CalibrationMethod.MovingAverage, values);
        }

        [Fact]
        public void Comparison_AreComparable()
        {
            // Arrange
            CalibrationMethod method1 = CalibrationMethod.MinMax;
            CalibrationMethod method2 = CalibrationMethod.Entropy;

            // Act & Assert
            Assert.True(method1 < method2);
            Assert.False(method1 > method2);
            Assert.True(method1 != method2);
            Assert.True(method2 > method1);
        }
    }
}
