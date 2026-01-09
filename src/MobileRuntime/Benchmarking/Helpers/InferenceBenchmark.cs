using System;
using System.Collections.Generic;
using System.Linq;
using MobileRuntime.Benchmarking.Models;
using MobileRuntime.Models;

namespace MobileRuntime.Benchmarking.Helpers;

public class InferenceBenchmark
{
    public static BenchmarkResult BenchmarkModel(
        IModel model,
        ITensor[] inputs,
        int iterations = 10,
        string name = "Inference")
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        using var runner = new BenchmarkRunner();

        var result = runner.RunBenchmark(name, () =>
        {
            model.Predict(inputs);
        }, iterations);

        return result;
    }

    public static BenchmarkResult BenchmarkOperator(
        IBackend backend,
        OperatorDescriptor op,
        ITensor[] inputs,
        int iterations = 100,
        string name = "Operator")
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        using var runner = new BenchmarkRunner();

        var result = runner.RunBenchmark(name, () =>
        {
            // Create output tensors
            var parameters = op.Parameters.Values.ToArray();
            var output = backend.ExecuteOperation(op.Type.ToString(), inputs, parameters);

            // Dispose output tensor to avoid memory leaks
            if (output != null)
            {
                (output as IDisposable)?.Dispose();
            }
        }, iterations);

        return result;
    }

    public static BenchmarkResults BenchmarkAllOperators(
        IBackend backend,
        Dictionary<OperatorType, OperatorDescriptor> operators,
        Dictionary<OperatorType, ITensor[]> inputs)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));
        if (operators == null)
            throw new ArgumentNullException(nameof(operators));
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        using var runner = new BenchmarkRunner();
        var benchmarkConfigs = new List<BenchmarkConfig>();

        foreach (var kvp in operators)
        {
            var opType = kvp.Key;
            var op = kvp.Value;

            if (inputs.TryGetValue(opType, out var opInputs))
            {
                benchmarkConfigs.Add(new BenchmarkConfig
                {
                    Name = $"{opType.ToString()}",
                    Benchmark = () =>
                    {
                        var parameters = op.Parameters.Values.ToArray();
                        var output = backend.ExecuteOperation(op.Type.ToString(), opInputs, parameters);
                        if (output != null)
                        {
                            (output as IDisposable)?.Dispose();
                        }
                    },
                    Iterations = 100
                });
            }
        }

        return runner.RunBenchmarkSuite("All Operators", benchmarkConfigs.ToArray());
    }
}
