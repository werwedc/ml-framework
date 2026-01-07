using System;
using Xunit;
using MLFramework.Quantization.DataStructures;

namespace MLFramework.Tests.Quantization.DataStructures
{
    public class QuantizationModeTests
    {
        [Fact]
        public void AllModeValues_AreValidEnumValues()
        {
            // Arrange & Act & Assert
            Assert.True(Enum.IsDefined(typeof(QuantizationMode), QuantizationMode.PerTensorSymmetric));
            Assert.True(Enum.IsDefined(typeof(QuantizationMode), QuantizationMode.PerTensorAsymmetric));
            Assert.True(Enum.IsDefined(typeof(QuantizationMode), QuantizationMode.PerChannelSymmetric));
            Assert.True(Enum.IsDefined(typeof(QuantizationMode), QuantizationMode.PerChannelAsymmetric));
        }

        [Theory]
        [InlineData(QuantizationMode.PerTensorSymmetric, "PerTensorSymmetric")]
        [InlineData(QuantizationMode.PerTensorAsymmetric, "PerTensorAsymmetric")]
        [InlineData(QuantizationMode.PerChannelSymmetric, "PerChannelSymmetric")]
        [InlineData(QuantizationMode.PerChannelAsymmetric, "PerChannelAsymmetric")]
        public void ToString_ReturnsCorrectStringRepresentation(QuantizationMode mode, string expected)
        {
            // Act
            string result = mode.ToString();

            // Assert
            Assert.Equal(expected, result);
        }

        [Theory]
        [InlineData("PerTensorSymmetric", QuantizationMode.PerTensorSymmetric)]
        [InlineData("PerTensorAsymmetric", QuantizationMode.PerTensorAsymmetric)]
        [InlineData("PerChannelSymmetric", QuantizationMode.PerChannelSymmetric)]
        [InlineData("PerChannelAsymmetric", QuantizationMode.PerChannelAsymmetric)]
        public void ParseString_ReturnsCorrectMode(string value, QuantizationMode expected)
        {
            // Act
            var result = (QuantizationMode)Enum.Parse(typeof(QuantizationMode), value);

            // Assert
            Assert.Equal(expected, result);
        }

        [Fact]
        public void ParseString_WithInvalidValue_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => Enum.Parse(typeof(QuantizationMode), "InvalidMode"));
        }

        [Theory]
        [InlineData("PER TENSORSYMMETRIC")]
        [InlineData("pertensorsymmetric")]
        [InlineData("PerTensorSymmetric ")]
        public void TryParseString_IsCaseInsensitive(string value)
        {
            // Act
            bool success = Enum.TryParse(value, true, out QuantizationMode result);

            // Assert
            Assert.True(success);
            Assert.Equal(QuantizationMode.PerTensorSymmetric, result);
        }

        [Fact]
        public void EnumValues_HaveSequentialIntegers()
        {
            // Act
            int perTensorSymmetric = (int)QuantizationMode.PerTensorSymmetric;
            int perTensorAsymmetric = (int)QuantizationMode.PerTensorAsymmetric;
            int perChannelSymmetric = (int)QuantizationMode.PerChannelSymmetric;
            int perChannelAsymmetric = (int)QuantizationMode.PerChannelAsymmetric;

            // Assert
            Assert.Equal(0, perTensorSymmetric);
            Assert.Equal(1, perTensorAsymmetric);
            Assert.Equal(2, perChannelSymmetric);
            Assert.Equal(3, perChannelAsymmetric);
        }

        [Theory]
        [InlineData(QuantizationMode.PerTensorSymmetric, true)]
        [InlineData(QuantizationMode.PerTensorAsymmetric, true)]
        [InlineData(QuantizationMode.PerChannelSymmetric, false)]
        [InlineData(QuantizationMode.PerChannelAsymmetric, false)]
        public void IsPerTensor_CorrectlyIdentifiesMode(QuantizationMode mode, bool expected)
        {
            // Act
            bool isPerTensor = mode == QuantizationMode.PerTensorSymmetric ||
                               mode == QuantizationMode.PerTensorAsymmetric;

            // Assert
            Assert.Equal(expected, isPerTensor);
        }

        [Theory]
        [InlineData(QuantizationMode.PerTensorSymmetric, true)]
        [InlineData(QuantizationMode.PerChannelSymmetric, true)]
        [InlineData(QuantizationMode.PerTensorAsymmetric, false)]
        [InlineData(QuantizationMode.PerChannelAsymmetric, false)]
        public void IsSymmetric_CorrectlyIdentifiesMode(QuantizationMode mode, bool expected)
        {
            // Act
            bool isSymmetric = mode == QuantizationMode.PerTensorSymmetric ||
                             mode == QuantizationMode.PerChannelSymmetric;

            // Assert
            Assert.Equal(expected, isSymmetric);
        }

        [Theory]
        [InlineData(QuantizationMode.PerTensorAsymmetric, true)]
        [InlineData(QuantizationMode.PerChannelAsymmetric, true)]
        [InlineData(QuantizationMode.PerTensorSymmetric, false)]
        [InlineData(QuantizationMode.PerChannelSymmetric, false)]
        public void IsAsymmetric_CorrectlyIdentifiesMode(QuantizationMode mode, bool expected)
        {
            // Act
            bool isAsymmetric = mode == QuantizationMode.PerTensorAsymmetric ||
                               mode == QuantizationMode.PerChannelAsymmetric;

            // Assert
            Assert.Equal(expected, isAsymmetric);
        }

        [Fact]
        public void GetNames_ReturnsAllModeNames()
        {
            // Act
            string[] names = Enum.GetNames(typeof(QuantizationMode));

            // Assert
            Assert.Equal(4, names.Length);
            Assert.Contains("PerTensorSymmetric", names);
            Assert.Contains("PerTensorAsymmetric", names);
            Assert.Contains("PerChannelSymmetric", names);
            Assert.Contains("PerChannelAsymmetric", names);
        }

        [Fact]
        public void GetValues_ReturnsAllModeValues()
        {
            // Act
            Array values = Enum.GetValues(typeof(QuantizationMode));

            // Assert
            Assert.Equal(4, values.Length);
            Assert.Contains(QuantizationMode.PerTensorSymmetric, values);
            Assert.Contains(QuantizationMode.PerTensorAsymmetric, values);
            Assert.Contains(QuantizationMode.PerChannelSymmetric, values);
            Assert.Contains(QuantizationMode.PerChannelAsymmetric, values);
        }
    }
}
