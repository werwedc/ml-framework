using System;
using Xunit;
using MLFramework.Communication.Optimizations;
using MLFramework.Distributed.Communication;

namespace MLFramework.Tests.Communication.Optimizations
{
    public class AlgorithmSelectorTests
    {
        private readonly CommunicationConfig _config;

        public AlgorithmSelectorTests()
        {
            _config = new CommunicationConfig();
        }

        [Fact]
        public void Constructor_NullConfig_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => new AlgorithmSelector(4, null));
        }

        [Fact]
        public void Constructor_ValidConfig_CreatesInstance()
        {
            var selector = new AlgorithmSelector(4, _config);
            Assert.NotNull(selector);
        }

        [Fact]
        public void SelectAllReduceAlgorithm_SmallMessage_ReturnsRecursiveDoubling()
        {
            var selector = new AlgorithmSelector(4, _config);
            var algorithm = selector.SelectAllReduceAlgorithm(2048);

            Assert.Equal(CommunicationAlgorithm.RecursiveDoubling, algorithm);
        }

        [Fact]
        public void SelectAllReduceAlgorithm_MediumMessage_ReturnsRabenseifner()
        {
            var selector = new AlgorithmSelector(8, _config);
            var algorithm = selector.SelectAllReduceAlgorithm(512 * 1024);

            Assert.Equal(CommunicationAlgorithm.Rabenseifner, algorithm);
        }

        [Fact]
        public void SelectAllReduceAlgorithm_LargeMessage_ReturnsRing()
        {
            var selector = new AlgorithmSelector(16, _config);
            var algorithm = selector.SelectAllReduceAlgorithm(8 * 1024 * 1024);

            Assert.Equal(CommunicationAlgorithm.Ring, algorithm);
        }

        [Fact]
        public void SelectAllReduceAlgorithm_VeryLargeMessage_ReturnsTree()
        {
            var selector = new AlgorithmSelector(16, _config);
            var algorithm = selector.SelectAllReduceAlgorithm(32 * 1024 * 1024);

            Assert.Equal(CommunicationAlgorithm.Tree, algorithm);
        }

        [Fact]
        public void SelectAllGatherAlgorithm_ReturnsRing()
        {
            var selector = new AlgorithmSelector(8, _config);
            var algorithm = selector.SelectAllGatherAlgorithm(1024 * 1024);

            Assert.Equal(CommunicationAlgorithm.Ring, algorithm);
        }

        [Fact]
        public void SelectReduceScatterAlgorithm_ReturnsRabenseifner()
        {
            var selector = new AlgorithmSelector(8, _config);
            var algorithm = selector.SelectReduceScatterAlgorithm(1024 * 1024);

            Assert.Equal(CommunicationAlgorithm.Rabenseifner, algorithm);
        }

        [Theory]
        [InlineData(CommunicationAlgorithm.Ring, "Ring")]
        [InlineData(CommunicationAlgorithm.Tree, "Tree")]
        [InlineData(CommunicationAlgorithm.RecursiveDoubling, "RecursiveDoubling")]
        [InlineData(CommunicationAlgorithm.Rabenseifner, "Rabenseifner")]
        [InlineData(CommunicationAlgorithm.Automatic, "Automatic")]
        public void GetAlgorithmName_ReturnsCorrectName(CommunicationAlgorithm algorithm, string expectedName)
        {
            var name = AlgorithmSelector.GetAlgorithmName(algorithm);
            Assert.Equal(expectedName, name);
        }
    }
}
