using System;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using MLFramework.Communication.Optimizations;

namespace MLFramework.Tests.Communication.Optimizations
{
    public class CommunicationProfilerTests : IDisposable
    {
        private readonly CommunicationProfiler _profiler;

        public CommunicationProfilerTests()
        {
            _profiler = new CommunicationProfiler();
        }

        [Fact]
        public void Constructor_CreatesInstance()
        {
            var profiler = new CommunicationProfiler();
            Assert.NotNull(profiler);
        }

        [Fact]
        public void Constructor_Disabled_DoesNotProfile()
        {
            var profiler = new CommunicationProfiler(false);
            var result = profiler.Profile("Test", 1024, () => 42);

            Assert.Equal(0, profiler.Profiles.Count);
            Assert.Equal(42, result);
        }

        [Fact]
        public void Profile_Function_ReturnsResult()
        {
            var result = _profiler.Profile("Test", 1024, () => 42);
            Assert.Equal(42, result);
        }

        [Fact]
        public void Profile_Function_CreatesProfile()
        {
            _profiler.Profile("Test", 1024, () => 42);

            Assert.Equal(1, _profiler.Profiles.Count);
        }

        [Fact]
        public void Profile_Function_SetsOperationName()
        {
            _profiler.Profile("TestOperation", 1024, () => 42);

            var profile = _profiler.Profiles.FirstOrDefault();
            Assert.Equal("TestOperation", profile?.Operation);
        }

        [Fact]
        public void Profile_Function_SetsDataSize()
        {
            _profiler.Profile("Test", 2048, () => 42);

            var profile = _profiler.Profiles.FirstOrDefault();
            Assert.Equal(2048, profile?.DataSizeBytes);
        }

        [Fact]
        public void Profile_Function_SetsDuration()
        {
            _profiler.Profile("Test", 1024, () => 42);

            var profile = _profiler.Profiles.FirstOrDefault();
            Assert.True(profile?.Duration.TotalMilliseconds > 0);
        }

        [Fact]
        public void Profile_Function_CalculatesBandwidth()
        {
            _profiler.Profile("Test", 1024 * 1024, () => 42);

            var profile = _profiler.Profiles.FirstOrDefault();
            Assert.True(profile?.BandwidthMBps > 0);
        }

        [Fact]
        public void Profile_Function_SetsAlgorithm()
        {
            _profiler.Profile("Test", 1024, () => 42, 0, "Ring");

            var profile = _profiler.Profiles.FirstOrDefault();
            Assert.Equal("Ring", profile?.Algorithm);
        }

        [Fact]
        public void Profile_AsyncFunction_ReturnsResult()
        {
            var result = _profiler.ProfileAsync("Test", 1024, () => Task.FromResult(42)).Result;
            Assert.Equal(42, result);
        }

        [Fact]
        public void Profile_AsyncFunction_CreatesProfile()
        {
            _profiler.ProfileAsync("Test", 1024, () => Task.FromResult(42)).Wait();

            Assert.Equal(1, _profiler.Profiles.Count);
        }

        [Fact]
        public void Profile_MultipleOperations_CreatesMultipleProfiles()
        {
            _profiler.Profile("Test1", 1024, () => 42);
            _profiler.Profile("Test2", 2048, () => 43);
            _profiler.Profile("Test3", 4096, () => 44);

            Assert.Equal(3, _profiler.Profiles.Count);
        }

        [Fact]
        public void Clear_RemovesAllProfiles()
        {
            _profiler.Profile("Test", 1024, () => 42);
            _profiler.Profile("Test", 1024, () => 43);

            Assert.Equal(2, _profiler.Profiles.Count);

            _profiler.Clear();
            Assert.Equal(0, _profiler.Profiles.Count);
        }

        [Fact]
        public void GetStatistics_Empty_ReturnsEmptyStatistics()
        {
            var stats = _profiler.GetStatistics();

            Assert.Equal(0, stats.TotalOperations);
            Assert.Equal(0, stats.TotalDataTransferred);
            Assert.Equal(0, stats.AverageBandwidth);
        }

        [Fact]
        public void GetStatistics_WithProfiles_ReturnsCorrectStatistics()
        {
            _profiler.Profile("Test", 1024, () => 42, 4, "Ring");
            _profiler.Profile("Test", 2048, () => 43, 4, "Ring");

            var stats = _profiler.GetStatistics();

            Assert.Equal(2, stats.TotalOperations);
            Assert.Equal(3072, stats.TotalDataTransferred);
            Assert.True(stats.AverageBandwidth > 0);
        }

        [Fact]
        public void GetStatistics_CalculatesMinAndMaxBandwidth()
        {
            _profiler.Profile("Test1", 1024, () => 42, 4, "Ring");
            _profiler.Profile("Test2", 2048, () => 43, 4, "Ring");

            var stats = _profiler.GetStatistics();

            Assert.True(stats.MinBandwidth > 0);
            Assert.True(stats.MaxBandwidth >= stats.MinBandwidth);
        }

        [Fact]
        public void Dispose_ClearsProfiles()
        {
            _profiler.Profile("Test", 1024, () => 42);
            Assert.Equal(1, _profiler.Profiles.Count);

            _profiler.Dispose();
            Assert.Equal(0, _profiler.Profiles.Count);
        }

        public void Dispose()
        {
            _profiler.Dispose();
        }
    }
}
