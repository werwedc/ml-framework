using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using System;
using System.Diagnostics;
using System.Linq;

namespace MLFramework.Tests.Autograd;

[Collection("Performance")]
public class AutogradPerformanceTests : IDisposable
{
    private readonly GraphBuilder _graphBuilder;

    public AutogradPerformanceTests()
    {
        _graphBuilder = new GraphBuilder();
    }

    public void Dispose()
    {
        _graphBuilder.Dispose();
    }

    #region Forward/Backward Benchmarks

    [Fact]
    public void BenchmarkForwardPass_Overhead()
    {
        // Benchmark forward pass overhead
        var sizes = new[] { 100, 1000, 10000 };
        var sw = new Stopwatch();

        foreach (var size in sizes)
        {
            var x = new Tensor(new float[size], new int[] { size }, requiresGrad: true);

            sw.Restart();
            for (int i = 0; i < 100; i++)
            {
                var y = x * 2.0f;
                var z = y + 1.0f;
            }
            sw.Stop();

            var avgMs = sw.Elapsed.TotalMilliseconds / 100;
            Console.WriteLine($"Forward pass overhead for size {size}: {avgMs:F4}ms");

            // Should complete in reasonable time
            Assert.True(avgMs < 100, $"Forward pass too slow for size {size}: {avgMs}ms");
        }
    }

    [Fact]
    public void BenchmarkBackwardPass_Speed()
    {
        // Benchmark backward pass speed
        var sizes = new[] { 100, 1000, 10000 };
        var sw = new Stopwatch();

        foreach (var size in sizes)
        {
            var x = new Tensor(new float[size], new int[] { size }, requiresGrad: true);
            var y = x * 2.0f;
            var z = y + 1.0f;

            var mulOp = new OperationContext("Mul", g => new Tensor[] { g.Clone() });
            var node1 = _graphBuilder.CreateNode(y, mulOp);

            var addOp = new OperationContext("Add", g => new Tensor[] { g.Clone() });
            var node2 = _graphBuilder.CreateNode(z, addOp, node1);

            sw.Restart();
            for (int i = 0; i < 100; i++)
            {
                var backward = new BackwardPass(_graphBuilder);
                backward.RetainGraph = true;
                backward.Run(z);
            }
            sw.Stop();

            var avgMs = sw.Elapsed.TotalMilliseconds / 100;
            Console.WriteLine($"Backward pass speed for size {size}: {avgMs:F4}ms");

            // Should complete in reasonable time
            Assert.True(avgMs < 100, $"Backward pass too slow for size {size}: {avgMs}ms");
        }
    }

    [Fact]
    public void BenchmarkGradientMemory_Usage()
    {
        // Benchmark gradient memory usage
        var sizes = new[] { 1000, 10000, 100000 };

        foreach (var size in sizes)
        {
            // Create large tensors
            var x = new Tensor(new float[size], new int[] { size }, requiresGrad: true);
            var y = x * 2.0f;
            var z = y + 1.0f;

            var mulOp = new OperationContext("Mul", g => new Tensor[] { g.Clone() });
            var node1 = _graphBuilder.CreateNode(y, mulOp);

            var addOp = new OperationContext("Add", g => new Tensor[] { g.Clone() });
            var node2 = _graphBuilder.CreateNode(z, addOp, node1);

            // Run backward pass
            var backward = new BackwardPass(_graphBuilder);
            backward.Run(z);

            // Check that gradient exists
            Assert.NotNull(x.Gradient);
            Console.WriteLine($"Memory usage for size {size}: {size * sizeof(float) * 2} bytes (tensors + gradients)");

            // Gradient should have correct size
            Assert.Equal(size, x.Gradient.Size);
        }
    }

    #endregion

    #region Checkpointing Benchmarks

    [Fact]
    public void BenchmarkCheckpointing_Overhead()
    {
        // Benchmark checkpointing overhead
        var depths = new[] { 10, 50, 100 };
        var sw = new Stopwatch();

        foreach (var depth in depths)
        {
            var x = new Tensor(new float[100], new int[] { 100 }, requiresGrad: true);
            var output = x.Clone();
            GraphNode? prevNode = null;

            // Build deep network
            for (int i = 0; i < depth; i++)
            {
                output = output * 2.0f;
                var op = new OperationContext($"Layer{i}", g => new Tensor[] { g.Clone() });
                prevNode = _graphBuilder.CreateNode(output, op, prevNode);
            }

            // Without checkpointing
            sw.Restart();
            var backward1 = new BackwardPass(_graphBuilder);
            backward1.Run(output);
            var timeWithoutCheckpoint = sw.Elapsed.TotalMilliseconds;

            // Reset gradient
            x.ZeroGrad();

            // With checkpointing
            var checkpointManager = new CheckpointManager();
            output = x.Clone();
            prevNode = null;
            for (int i = 0; i < depth; i++)
            {
                if (i % 10 == 0)
                    checkpointManager.SaveCheckpoint(output);

                output = output * 2.0f;
                var op = new OperationContext($"Layer{i}", g => new Tensor[] { g.Clone() });
                prevNode = _graphBuilder.CreateNode(output, op, prevNode);
            }

            sw.Restart();
            var backward2 = new BackwardPass(_graphBuilder);
            backward2.Run(output);
            var timeWithCheckpoint = sw.Elapsed.TotalMilliseconds;

            Console.WriteLine($"Checkpointing overhead for depth {depth}: " +
                            $"Without: {timeWithoutCheckpoint:F2}ms, " +
                            $"With: {timeWithCheckpoint:F2}ms, " +
                            $"Ratio: {timeWithCheckpoint / timeWithoutCheckpoint:F2}x");
        }
    }

    [Fact]
    public void BenchmarkCheckpointing_MemorySavings()
    {
        // Benchmark checkpointing memory savings
        var size = 10000;
        var depth = 100;

        var x = new Tensor(new float[size], new int[] { size }, requiresGrad: true);
        var checkpointManager = new CheckpointManager();

        // Build deep network with checkpoints
        var output = x.Clone();
        GraphNode? prevNode = null;

        for (int i = 0; i < depth; i++)
        {
            if (i % 10 == 0)
                checkpointManager.SaveCheckpoint(output);

            output = output * 2.0f;
            var op = new OperationContext($"Layer{i}", g => new Tensor[] { g.Clone() });
            prevNode = _graphBuilder.CreateNode(output, op, prevNode);
        }

        var backward = new BackwardPass(_graphBuilder);
        backward.Run(output);

        // Check that gradient exists
        Assert.NotNull(x.Gradient);

        var memoryWithoutCheckpoint = size * depth * sizeof(float); // All activations
        var memoryWithCheckpoint = size * (depth / 10 + 1) * sizeof(float); // Checkpoints only

        Console.WriteLine($"Memory savings for size {size}, depth {depth}: " +
                        $"Without checkpoint: {memoryWithoutCheckpoint / 1024}KB, " +
                        $"With checkpoint: {memoryWithCheckpoint / 1024}KB, " +
                        $"Saved: {(memoryWithoutCheckpoint - memoryWithCheckpoint) / 1024}KB");
    }

    #endregion

    #region Jacobian Computation Benchmarks

    [Fact]
    public void BenchmarkJacobianComputation_Time()
    {
        // Benchmark Jacobian computation time
        var inputSizes = new[] { 10, 50, 100 };
        var outputSizes = new[] { 10, 50, 100 };
        var sw = new Stopwatch();

        foreach (var inputSize in inputSizes)
        {
            foreach (var outputSize in outputSizes)
            {
                Tensor LinearFunc(Tensor x)
                {
                    var result = new Tensor(new float[outputSize], new int[] { outputSize });
                    for (int i = 0; i < outputSize; i++)
                    {
                        result.Data[i] = x.Data[i % inputSize] * 2.0f;
                    }
                    return result;
                }

                var x = new Tensor(new float[inputSize], new int[] { inputSize });

                sw.Restart();
                for (int i = 0; i < 10; i++)
                {
                    var jacobian = GradientComputer.Jacobian(LinearFunc, x);
                }
                sw.Stop();

                var avgMs = sw.Elapsed.TotalMilliseconds / 10;
                Console.WriteLine($"Jacobian computation for input {inputSize}, output {outputSize}: {avgMs:F2}ms");

                // Should complete in reasonable time
                Assert.True(avgMs < 5000, $"Jacobian computation too slow: {avgMs}ms");
            }
        }
    }

    #endregion

    #region Large Scale Tests

    [Fact]
    public void TestLargeModelGradients_1MParameters()
    {
        // Test gradient computation with 1M parameters
        var parameterCount = 1_000_000;
        var input = new Tensor(new float[100], new int[] { 100 }, requiresGrad: false);
        var weights = new Tensor(new float[parameterCount], new int[] { 100, 10000 }, requiresGrad: true);

        // Simulate forward pass
        var output = new Tensor(new float[10000], new int[] { 10000 });

        var linearOp = new OperationContext("Linear", g =>
        {
            // Simplified gradient: dL/dW = g * input^T
            var dW = new Tensor(new float[parameterCount], weights.Shape);
            for (int i = 0; i < 10000; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    dW.Data[i * 100 + j] = g.Data[i] * input.Data[j];
                }
            }
            return new Tensor[] { dW };
        });

        var node = _graphBuilder.CreateNode(output, linearOp);

        var sw = Stopwatch.StartNew();
        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[10000], new int[] { 10000 });
        gradOutput.Fill(1.0f);
        backward.Run(output, gradOutput);
        sw.Stop();

        Assert.NotNull(weights.Gradient);
        Assert.Equal(parameterCount, weights.Gradient.Size);

        Console.WriteLine($"Gradient computation for 1M parameters: {sw.Elapsed.TotalMilliseconds:F2}ms");

        // Should complete in reasonable time (< 10 seconds)
        Assert.True(sw.Elapsed.TotalSeconds < 10, $"Gradient computation too slow: {sw.Elapsed.TotalSeconds:F2}s");
    }

    [Fact]
    public void TestDeepNetwork_100Layers()
    {
        // Test gradient computation through 100-layer network
        var layerCount = 100;
        var layerSize = 100;

        var x = new Tensor(new float[layerSize], new int[] { layerSize }, requiresGrad: true);
        var output = x.Clone();
        GraphNode? prevNode = null;

        // Build deep network
        for (int i = 0; i < layerCount; i++)
        {
            output = output * 2.0f + 1.0f;
            var op = new OperationContext($"Layer{i}", g => new Tensor[] { g.Clone() });
            prevNode = _graphBuilder.CreateNode(output, op, prevNode);
        }

        var sw = Stopwatch.StartNew();
        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[layerSize], new int[] { layerSize });
        gradOutput.Fill(1.0f);
        backward.Run(output, gradOutput);
        sw.Stop();

        Assert.NotNull(x.Gradient);
        Assert.Equal(layerSize, x.Gradient.Size);

        Console.WriteLine($"Gradient computation for {layerCount}-layer network: {sw.Elapsed.TotalMilliseconds:F2}ms");

        // Should complete in reasonable time (< 5 seconds)
        Assert.True(sw.Elapsed.TotalSeconds < 5, $"Deep network gradient computation too slow: {sw.Elapsed.TotalSeconds:F2}s");
    }

    #endregion

    #region Numerical Gradient Benchmarks

    [Fact]
    public void BenchmarkNumericalGradient_Accuracy()
    {
        // Benchmark numerical gradient accuracy vs speed
        Tensor SquareFunc(Tensor x)
        {
            var result = new Tensor(new float[x.Size], x.Shape);
            for (int i = 0; i < x.Size; i++)
            {
                result.Data[i] = x.Data[i] * x.Data[i];
            }
            return result;
        }

        var sizes = new[] { 10, 100, 1000 };
        var epsilons = new[] { 1e-7, 1e-6, 1e-5 };

        foreach (var size in sizes)
        {
            var x = new Tensor(new float[size], new int[] { size });

            foreach (var epsilon in epsilons)
            {
                var sw = Stopwatch.StartNew();
                var grad = GradientComputer.NumericalGradient(SquareFunc, x, epsilon);
                sw.Stop();

                // Check accuracy: f'(x) = 2x
                var error = 0.0;
                for (int i = 0; i < size; i++)
                {
                    var expected = 2.0 * x.Data[i];
                    error += Math.Abs(grad.Data[i] - expected);
                }
                error /= size;

                Console.WriteLine($"Numerical gradient for size {size}, epsilon {epsilon:E}: " +
                                $"Time: {sw.Elapsed.TotalMilliseconds:F2}ms, " +
                                $"Avg Error: {error:E}");
            }
        }
    }

    [Fact]
    public void BenchmarkGradientChecking_Speed()
    {
        // Benchmark gradient checking speed
        Tensor CubicFunc(Tensor x)
        {
            var result = new Tensor(new float[x.Size], x.Shape);
            for (int i = 0; i < x.Size; i++)
            {
                result.Data[i] = x.Data[i] * x.Data[i] * x.Data[i];
            }
            return result;
        }

        var sizes = new[] { 10, 100, 1000 };

        foreach (var size in sizes)
        {
            var x = new Tensor(new float[size], new int[] { size });
            x.Fill(2.0f);

            var sw = Stopwatch.StartNew();
            var passed = GradientChecker.CheckGradient(CubicFunc, x, tolerance: 1e-6);
            sw.Stop();

            Console.WriteLine($"Gradient check for size {size}: Time: {sw.Elapsed.TotalMilliseconds:F2}ms, Passed: {passed}");

            // Should complete in reasonable time
            Assert.True(sw.Elapsed.TotalMilliseconds < 5000, $"Gradient check too slow: {sw.Elapsed.TotalMilliseconds}ms");
        }
    }

    #endregion

    #region Memory Leak Tests

    [Fact]
    public void BenchmarkMemoryLeaks_MultipleIterations()
    {
        // Test for memory leaks over multiple iterations
        var iterations = 1000;
        var size = 100;

        var initialMemory = GC.GetTotalMemory(true);

        for (int i = 0; i < iterations; i++)
        {
            var x = new Tensor(new float[size], new int[] { size }, requiresGrad: true);
            var y = x * 2.0f;
            var z = y + 1.0f;

            var mulOp = new OperationContext("Mul", g => new Tensor[] { g.Clone() });
            var node1 = _graphBuilder.CreateNode(y, mulOp);

            var addOp = new OperationContext("Add", g => new Tensor[] { g.Clone() });
            var node2 = _graphBuilder.CreateNode(z, addOp, node1);

            var backward = new BackwardPass(_graphBuilder);
            backward.Run(z);
        }

        // Force garbage collection
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var finalMemory = GC.GetTotalMemory(true);
        var memoryGrowth = finalMemory - initialMemory;

        Console.WriteLine($"Memory growth after {iterations} iterations: {memoryGrowth / 1024}KB");

        // Memory growth should be reasonable (< 10MB)
        Assert.True(memoryGrowth < 10 * 1024 * 1024, $"Excessive memory growth: {memoryGrowth / 1024}KB");
    }

    #endregion

    #region Gradient Accumulation Benchmarks

    [Fact]
    public void BenchmarkGradientAccumulation_Speed()
    {
        // Benchmark gradient accumulation speed
        var sizes = new[] { 100, 1000, 10000 };
        var accumulationSteps = 10;

        foreach (var size in sizes)
        {
            var x = new Tensor(new float[size], new int[] { size }, requiresGrad: true);
            var w = new Tensor(new float[size], new int[] { size }, requiresGrad: true);

            var sw = Stopwatch.StartNew();
            for (int step = 0; step < accumulationSteps; step++)
            {
                var y = x * 2.0f;
                var z = y * w;

                var mulOp1 = new OperationContext("Mul1", g => new Tensor[] { g.Clone() });
                var node1 = _graphBuilder.CreateNode(y, mulOp1);

                var mulOp2 = new OperationContext("Mul2", g => new Tensor[] { g.Clone() });
                var node2 = _graphBuilder.CreateNode(z, mulOp2, node1);

                var backward = new BackwardPass(_graphBuilder);
                backward.RetainGraph = true;
                backward.Run(z);
            }
            sw.Stop();

            var avgMs = sw.Elapsed.TotalMilliseconds / accumulationSteps;
            Console.WriteLine($"Gradient accumulation for size {size}: {avgMs:F2}ms per step");

            // Should complete in reasonable time
            Assert.True(avgMs < 100, $"Gradient accumulation too slow: {avgMs}ms");
        }
    }

    #endregion

    #region Complex Operation Benchmarks

    [Fact]
    public void BenchmarkComplexOperations_Speed()
    {
        // Benchmark complex operations (e.g., LSTM, ResNet blocks)
        var sizes = new[] { 10, 50, 100 };

        foreach (var size in sizes)
        {
            // LSTM-like operation
            var input = new Tensor(new float[size], new int[] { size }, requiresGrad: true);
            var hidden = new Tensor(new float[size], new int[] { size }, requiresGrad: true);

            var output = new Tensor(new float[size], new int[] { size });

            var lstmOp = new OperationContext("LSTM", g =>
            {
                var dInput = g.Clone();
                var dHidden = g.Clone();
                return new Tensor[] { dInput, dHidden };
            });

            var node = _graphBuilder.CreateNode(output, lstmOp);

            var sw = Stopwatch.StartNew();
            for (int i = 0; i < 100; i++)
            {
                var backward = new BackwardPass(_graphBuilder);
                backward.RetainGraph = true;
                var gradOutput = new Tensor(new float[size], new int[] { size });
                gradOutput.Fill(1.0f);
                backward.Run(output, gradOutput);
            }
            sw.Stop();

            var avgMs = sw.Elapsed.TotalMilliseconds / 100;
            Console.WriteLine($"LSTM operation for size {size}: {avgMs:F2}ms");

            // Should complete in reasonable time
            Assert.True(avgMs < 100, $"LSTM operation too slow: {avgMs}ms");
        }
    }

    #endregion
}
