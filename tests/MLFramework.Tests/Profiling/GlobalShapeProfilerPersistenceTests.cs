using System;
using System.IO;
using MLFramework.Profiling;
using Xunit;

namespace MLFramework.Tests.Profiling
{
    public class GlobalShapeProfilerPersistenceTests
    {
        private readonly string _testPath;

        public GlobalShapeProfilerPersistenceTests()
        {
            _testPath = Path.Combine(Path.GetTempPath(), $"shape_profiler_test_{Guid.NewGuid()}.json");
        }

        public void Dispose()
        {
            if (File.Exists(_testPath))
            {
                File.Delete(_testPath);
            }
        }

        [Fact]
        public void Persist_SavesProfileToFile()
        {
            var profiler = new GlobalShapeProfiler();
            profiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });
            profiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });
            profiler.RecordShape("tensor2", "mul", new[] { 5, 10, 15 });

            profiler.Persist(_testPath);

            Assert.True(File.Exists(_testPath));
        }

        [Fact]
        public void Persist_ThrowsWithoutPath()
        {
            var profiler = new GlobalShapeProfiler();
            profiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });

            Assert.Throws<InvalidOperationException>(() => profiler.Persist());
        }

        [Fact]
        public void Load_RestoresProfileFromFile()
        {
            var originalProfiler = new GlobalShapeProfiler(maxSamplesPerTensor: 5000);
            originalProfiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });
            originalProfiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });
            originalProfiler.RecordShape("tensor2", "mul", new[] { 5, 10, 15 });

            originalProfiler.Persist(_testPath);

            var loadedProfiler = new GlobalShapeProfiler();
            loadedProfiler.Load(_testPath);

            Assert.Equal(2, loadedProfiler.TensorHistograms.Count);
            Assert.Equal(5000, loadedProfiler.MaxSamplesPerTensor);

            var histogram1 = loadedProfiler.GetHistogram("tensor1");
            Assert.NotNull(histogram1);
            Assert.Equal(2, histogram1.TotalSamples);

            var histogram2 = loadedProfiler.GetHistogram("tensor2");
            Assert.NotNull(histogram2);
            Assert.Equal(1, histogram2.TotalSamples);
        }

        [Fact]
        public void Load_ThrowsOnNonExistentFile()
        {
            var profiler = new GlobalShapeProfiler();

            Assert.Throws<FileNotFoundException>(() => profiler.Load("non_existent_file.json"));
        }

        [Fact]
        public void PersistAndLoad_RoundTrip()
        {
            var originalProfiler = new GlobalShapeProfiler();
            originalProfiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });
            originalProfiler.RecordShape("tensor1", "mul", new[] { 15, 25, 35 });
            originalProfiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });

            var report = originalProfiler.GetReport();

            originalProfiler.Persist(_testPath);

            var loadedProfiler = new GlobalShapeProfiler();
            loadedProfiler.Load(_testPath);

            var loadedReport = loadedProfiler.GetReport();

            Assert.Equal(report, loadedReport);
        }

        [Fact]
        public void Load_CreatesDirectoryIfNeeded()
        {
            var pathWithDir = Path.Combine(Path.GetTempPath(), "test_dir", "subdir", "profile.json");

            var profiler = new GlobalShapeProfiler();
            profiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });

            profiler.Persist(pathWithDir);

            Assert.True(File.Exists(pathWithDir));

            // Cleanup
            File.Delete(pathWithDir);
            Directory.Delete(Path.GetDirectoryName(pathWithDir)!);
            Directory.Delete(Path.GetDirectoryName(Path.GetDirectoryName(pathWithDir))!);
        }
    }
}
