using BenchmarkDotNet.Running;
using MlFramework.Benchmarks.Inference.PagedAttention;

namespace MlFramework.Benchmarks;

class Program
{
    static void Main(string[] args)
    {
        var switcher = new BenchmarkSwitcher(new[]
        {
            typeof(PagedAttentionBenchmarks),
            typeof(BlockManagerBenchmarks),
            typeof(BlockTableBenchmarks),
            typeof(ServingWorkloadBenchmark)
        });

        switcher.Run(args);
    }
}
