using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using MLFramework.Data.Metrics;
using Xunit;

namespace MLFramework.Tests.Data.Metrics
{
    public class ConsoleExporterTests
    {
        [Fact]
        public void Export_WithNullMetrics_ThrowsArgumentNullException()
        {
            var exporter = new ConsoleExporter();

            Assert.Throws<ArgumentNullException>(() => exporter.Export(null));
        }

        [Fact]
        public void Export_WithEmptyMetrics_DoesNotThrow()
        {
            var exporter = new ConsoleExporter();
            var metrics = new Dictionary<string, VersionMetrics>();

            var exception = Record.Exception(() => exporter.Export(metrics));

            Assert.Null(exception);
        }

        [Fact]
        public void Export_WithValidMetrics_OutputsToConsole()
        {
            var metrics = CreateTestMetrics();
            var exporter = new ConsoleExporter();

            var exception = Record.Exception(() => exporter.Export(metrics));

            Assert.Null(exception);
        }

        [Fact]
        public async Task ExportAsync_WithValidMetrics_OutputsToConsole()
        {
            var metrics = CreateTestMetrics();
            var exporter = new ConsoleExporter();

            var exception = await Record.ExceptionAsync(() => exporter.ExportAsync(metrics));

            Assert.Null(exception);
        }

        private Dictionary<string, VersionMetrics> CreateTestMetrics()
        {
            var metrics = new VersionMetrics(
                "model1",
                "v1",
                DateTime.UtcNow.AddMinutes(-5),
                DateTime.UtcNow,
                100,
                10.5,
                50.0,
                45.0,
                90.0,
                99.0,
                2.5,
                42,
                512.0,
                2,
                new Dictionary<string, long> { { "Timeout", 1 }, { "OOM", 1 } }
            );

            return new Dictionary<string, VersionMetrics>
            {
                { "model1:v1", metrics }
            };
        }
    }

    public class PrometheusExporterTests
    {
        [Fact]
        public void Constructor_WithNullOutput_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => new PrometheusExporter(null));
        }

        [Fact]
        public void Constructor_WithValidOutput_InitializesCorrectly()
        {
            using var stringWriter = new StringWriter();
            var exporter = new PrometheusExporter(stringWriter);

            Assert.NotNull(exporter);
        }

        [Fact]
        public void Export_WithNullMetrics_ThrowsArgumentNullException()
        {
            using var stringWriter = new StringWriter();
            var exporter = new PrometheusExporter(stringWriter);

            Assert.Throws<ArgumentNullException>(() => exporter.Export(null));
        }

        [Fact]
        public void Export_WithValidMetrics_OutputsPrometheusFormat()
        {
            using var stringWriter = new StringWriter();
            var exporter = new PrometheusExporter(stringWriter);
            var metrics = CreateTestMetrics();

            exporter.Export(metrics);

            var output = stringWriter.ToString();

            Assert.Contains("# HELP model_inference_requests_total", output);
            Assert.Contains("# TYPE model_inference_requests_total counter", output);
            Assert.Contains("model_inference_requests_total{model_name=\"model1\",version=\"v1\"} 100", output);
            Assert.Contains("model_inference_latency_ms", output);
            Assert.Contains("model_memory_usage_bytes", output);
            Assert.Contains("model_active_connections", output);
        }

        [Fact]
        public void Export_IncludesHelpAndTypeDefinitions()
        {
            using var stringWriter = new StringWriter();
            var exporter = new PrometheusExporter(stringWriter);
            var metrics = CreateTestMetrics();

            exporter.Export(metrics);

            var output = stringWriter.ToString();

            // Check for HELP definitions
            Assert.Contains("# HELP model_inference_requests_total Total number of inference requests", output);
            Assert.Contains("# HELP model_inference_latency_ms Inference latency in milliseconds", output);
            Assert.Contains("# HELP model_inference_errors_total Total number of inference errors", output);
            Assert.Contains("# HELP model_memory_usage_bytes Memory usage in bytes", output);
            Assert.Contains("# HELP model_active_connections Number of active connections", output);

            // Check for TYPE definitions
            Assert.Contains("# TYPE model_inference_requests_total counter", output);
            Assert.Contains("# TYPE model_inference_latency_ms histogram", output);
            Assert.Contains("# TYPE model_inference_errors_total counter", output);
            Assert.Contains("# TYPE model_memory_usage_bytes gauge", output);
            Assert.Contains("# TYPE model_active_connections gauge", output);
        }

        [Fact]
        public void Export_IncludesLatencyPercentiles()
        {
            using var stringWriter = new StringWriter();
            var exporter = new PrometheusExporter(stringWriter);
            var metrics = CreateTestMetrics();

            exporter.Export(metrics);

            var output = stringWriter.ToString();

            Assert.Contains("quantile=\"0.5\"", output);
            Assert.Contains("quantile=\"0.95\"", output);
            Assert.Contains("quantile=\"0.99\"", output);
        }

        [Fact]
        public void Export_IncludesErrorBreakdownByType()
        {
            using var stringWriter = new StringWriter();
            var exporter = new PrometheusExporter(stringWriter);
            var metrics = CreateTestMetrics();

            exporter.Export(metrics);

            var output = stringWriter.ToString();

            Assert.Contains("error_type=\"Timeout\"", output);
            Assert.Contains("error_type=\"OOM\"", output);
        }

        [Fact]
        public async Task ExportAsync_WithValidMetrics_OutputsPrometheusFormat()
        {
            using var stringWriter = new StringWriter();
            var exporter = new PrometheusExporter(stringWriter);
            var metrics = CreateTestMetrics();

            await exporter.ExportAsync(metrics);

            var output = stringWriter.ToString();

            Assert.Contains("# HELP model_inference_requests_total", output);
            Assert.Contains("model_inference_requests_total{model_name=\"model1\",version=\"v1\"} 100", output);
        }

        private Dictionary<string, VersionMetrics> CreateTestMetrics()
        {
            var metrics = new VersionMetrics(
                "model1",
                "v1",
                DateTime.UtcNow.AddMinutes(-5),
                DateTime.UtcNow,
                100,
                10.5,
                50.0,
                45.0,
                90.0,
                99.0,
                2.5,
                42,
                512.0,
                2,
                new Dictionary<string, long> { { "Timeout", 1 }, { "OOM", 1 } }
            );

            return new Dictionary<string, VersionMetrics>
            {
                { "model1:v1", metrics }
            };
        }
    }

    public class StatsDExporterTests
    {
        [Fact]
        public void Constructor_Default_InitializesWithConsoleOutput()
        {
            using var exporter = new StatsDExporter();

            Assert.NotNull(exporter);
        }

        [Fact]
        public void Constructor_WithHostAndPort_InitializesUdpClient()
        {
            using var exporter = new StatsDExporter("localhost", 8125);

            Assert.NotNull(exporter);
        }

        [Fact]
        public void Constructor_WithPrefix_InitializesCorrectly()
        {
            using var exporter = new StatsDExporter(prefix: "test");

            Assert.NotNull(exporter);
        }

        [Fact]
        public void Export_WithNullMetrics_ThrowsArgumentNullException()
        {
            using var exporter = new StatsDExporter();

            Assert.Throws<ArgumentNullException>(() => exporter.Export(null));
        }

        [Fact]
        public void Export_WithConsoleOutput_OutputsToConsole()
        {
            using var stringWriter = new StringWriter();
            Console.SetOut(stringWriter);

            using var exporter = new StatsDExporter();
            var metrics = CreateTestMetrics();

            var exception = Record.Exception(() => exporter.Export(metrics));

            Assert.Null(exception);
        }

        [Fact]
        public void Export_IncludesCounterMetrics()
        {
            using var stringWriter = new StringWriter();
            Console.SetOut(stringWriter);

            using var exporter = new StatsDExporter();
            var metrics = CreateTestMetrics();

            exporter.Export(metrics);

            var output = stringWriter.ToString();

            Assert.Contains("|c", output); // Counter type
        }

        [Fact]
        public void Export_IncludesTimingMetrics()
        {
            using var stringWriter = new StringWriter();
            Console.SetOut(stringWriter);

            using var exporter = new StatsDExporter();
            var metrics = CreateTestMetrics();

            exporter.Export(metrics);

            var output = stringWriter.ToString();

            Assert.Contains("|ms", output); // Timing type
        }

        [Fact]
        public void Export_IncludesGaugeMetrics()
        {
            using var stringWriter = new StringWriter();
            Console.SetOut(stringWriter);

            using var exporter = new StatsDExporter();
            var metrics = CreateTestMetrics();

            exporter.Export(metrics);

            var output = stringWriter.ToString();

            Assert.Contains("|g", output); // Gauge type
        }

        [Fact]
        public void Export_IncludesLatencyPercentiles()
        {
            using var stringWriter = new StringWriter();
            Console.SetOut(stringWriter);

            using var exporter = new StatsDExporter();
            var metrics = CreateTestMetrics();

            exporter.Export(metrics);

            var output = stringWriter.ToString();

            Assert.Contains(".p50|", output);
            Assert.Contains(".p95|", output);
            Assert.Contains(".p99|", output);
        }

        [Fact]
        public void Export_WithPrefix_IncludesPrefixInMetricNames()
        {
            using var stringWriter = new StringWriter();
            Console.SetOut(stringWriter);

            using var exporter = new StatsDExporter(prefix: "myapp");
            var metrics = CreateTestMetrics();

            exporter.Export(metrics);

            var output = stringWriter.ToString();

            Assert.Contains("myapp.model", output);
        }

        [Fact]
        public void Export_SanitizesMetricNames()
        {
            using var stringWriter = new StringWriter();
            Console.SetOut(stringWriter);

            using var exporter = new StatsDExporter();
            var metrics = CreateTestMetricsWithSpecialChars();

            exporter.Export(metrics);

            var output = stringWriter.ToString();

            // Special characters should be replaced with underscores
            Assert.DoesNotContain("model-name", output);
            Assert.DoesNotContain("model-name", output);
        }

        [Fact]
        public void Export_IncludesErrorBreakdownByType()
        {
            using var stringWriter = new StringWriter();
            Console.SetOut(stringWriter);

            using var exporter = new StatsDExporter();
            var metrics = CreateTestMetrics();

            exporter.Export(metrics);

            var output = stringWriter.ToString();

            Assert.Contains(".errors.Timeout|", output);
            Assert.Contains(".errors.OOM|", output);
        }

        [Fact]
        public async Task ExportAsync_WithValidMetrics_ExportsSuccessfully()
        {
            using var stringWriter = new StringWriter();
            Console.SetOut(stringWriter);

            using var exporter = new StatsDExporter();
            var metrics = CreateTestMetrics();

            var exception = await Record.ExceptionAsync(() => exporter.ExportAsync(metrics));

            Assert.Null(exception);
        }

        [Fact]
        public void Dispose_CanBeCalledMultipleTimes()
        {
            var exporter = new StatsDExporter();

            exporter.Dispose();
            exporter.Dispose(); // Should not throw
        }

        private Dictionary<string, VersionMetrics> CreateTestMetrics()
        {
            var metrics = new VersionMetrics(
                "model1",
                "v1",
                DateTime.UtcNow.AddMinutes(-5),
                DateTime.UtcNow,
                100,
                10.5,
                50.0,
                45.0,
                90.0,
                99.0,
                2.5,
                42,
                512.0,
                2,
                new Dictionary<string, long> { { "Timeout", 1 }, { "OOM", 1 } }
            );

            return new Dictionary<string, VersionMetrics>
            {
                { "model1:v1", metrics }
            };
        }

        private Dictionary<string, VersionMetrics> CreateTestMetricsWithSpecialChars()
        {
            var metrics = new VersionMetrics(
                "model-name",
                "v-1.0",
                DateTime.UtcNow.AddMinutes(-5),
                DateTime.UtcNow,
                100,
                10.5,
                50.0,
                45.0,
                90.0,
                99.0,
                2.5,
                42,
                512.0,
                2,
                new Dictionary<string, long>()
            );

            return new Dictionary<string, VersionMetrics>
            {
                { "model-name:v-1.0", metrics }
            };
        }
    }
}
