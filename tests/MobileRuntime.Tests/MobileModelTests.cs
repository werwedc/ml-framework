using Microsoft.VisualStudio.TestTools.UnitTesting;
using FluentAssertions;

namespace MobileRuntime.Tests
{
    [TestClass]
    public class MobileModelTests
    {
        [TestMethod]
        public void RuntimeMobileRuntime_LoadModel_WithValidPath_ReturnsModel()
        {
            // Arrange
            var runtime = new RuntimeMobileRuntime();

            // Act
            var model = runtime.LoadModel("test_model.mob");

            // Assert
            model.Should().NotBeNull();
            model.Name.Should().Be("test_model.mob");
        }

        [TestMethod]
        public void RuntimeMobileRuntime_LoadModel_WithValidBytes_ReturnsModel()
        {
            // Arrange
            var runtime = new RuntimeMobileRuntime();
            var modelBytes = new byte[] { 1, 2, 3, 4 };

            // Act
            var model = runtime.LoadModel(modelBytes);

            // Assert
            model.Should().NotBeNull();
            model.Name.Should().Be("LoadedModel");
        }

        [TestMethod]
        public void Model_Inputs_ReturnsExpectedInfo()
        {
            // Arrange
            var runtime = new RuntimeMobileRuntime();
            var model = runtime.LoadModel("test_model.mob");

            // Act
            var inputs = model.Inputs;

            // Assert
            inputs.Should().NotBeNull();
            inputs.Should().HaveCount(1);
            inputs[0].Name.Should().Be("input");
        }

        [TestMethod]
        public void Model_Outputs_ReturnsExpectedInfo()
        {
            // Arrange
            var runtime = new RuntimeMobileRuntime();
            var model = runtime.LoadModel("test_model.mob");

            // Act
            var outputs = model.Outputs;

            // Assert
            outputs.Should().NotBeNull();
            outputs.Should().HaveCount(1);
            outputs[0].Name.Should().Be("output");
        }
    }
}
