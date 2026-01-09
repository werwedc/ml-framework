using Microsoft.VisualStudio.TestTools.UnitTesting;
using FluentAssertions;

namespace MobileRuntime.Tests
{
    [TestClass]
    public class IntegrationTests
    {
        [TestMethod]
        public void Runtime_LoadModel_And_Predict_Works()
        {
            // Arrange
            var runtime = new RuntimeMobileRuntime();
            var input = new Tensor
            {
                Data = new[] { 1.0f, 2.0f, 3.0f, 4.0f },
                Shape = new[] { 2, 2 },
                DataType = DataType.Float32
            };

            // Act
            var model = runtime.LoadModel("test_model.mob");
            var outputs = model.Predict(new[] { input });

            // Assert
            model.Should().NotBeNull();
            outputs.Should().NotBeNull();
            outputs.Should().HaveCount(1);
        }

        [TestMethod]
        public void Runtime_GetRuntimeInfo_ReturnsValidInfo()
        {
            // Arrange
            var runtime = new RuntimeMobileRuntime();

            // Act
            var info = runtime.GetRuntimeInfo();

            // Assert
            info.Should().NotBeNull();
            info.Version.Should().NotBeNullOrEmpty();
            info.SupportedBackends.Should().NotBeEmpty();
        }
    }
}
