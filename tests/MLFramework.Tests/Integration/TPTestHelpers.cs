using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Modules;
using MLFramework.Optimizers;
using MLFramework.Tensor;
using MLFramework.Distributed;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Tests.Integration
{
    /// <summary>
    /// Test helper utilities for Tensor Parallelism tests
    /// </summary>
    public static class TPTestHelpers
    {
        /// <summary>
        /// Create a mock TP context for testing
        /// </summary>
        public static MockTensorParallelContext CreateMockTPContext(int worldSize, int rank)
        {
            return new MockTensorParallelContext(worldSize, rank);
        }

        /// <summary>
        /// Create a simple TP MLP model
        /// </summary>
        public static Module CreateSimpleTPMLP(
            int inputSize,
            int hiddenSize,
            int outputSize,
            bool bias = true,
            MockTensorParallelContext context = null)
        {
            var layers = new Sequential();

            // Column-parallel: input → hidden
            var fc1 = new MockColumnParallelLinear(
                inputSize, hiddenSize, bias: bias, gatherOutput: false, context: context);
            layers.AddModule("fc1", fc1);
            layers.AddModule("relu1", new ReLU());

            // Row-parallel: hidden → output
            var fc2 = new MockRowParallelLinear(
                hiddenSize, outputSize, bias: bias, inputIsSharded: true, context: context);
            layers.AddModule("fc2", fc2);

            return layers;
        }

        /// <summary>
        /// Create a standard (non-parallel) MLP for comparison
        /// </summary>
        public static Module CreateStandardMLP(
            int inputSize,
            int hiddenSize,
            int outputSize,
            bool bias = true)
        {
            var layers = new Sequential();

            var fc1 = new LinearLayer(inputSize, hiddenSize, bias: bias);
            layers.AddModule("fc1", fc1);
            layers.AddModule("relu1", new ReLU());

            var fc2 = new LinearLayer(hiddenSize, outputSize, bias: bias);
            layers.AddModule("fc2", fc2);

            return layers;
        }

        /// <summary>
        /// Create test data
        /// </summary>
        public static Tensor CreateTestInput(int batchSize, int inputSize, int seed = 42)
        {
            var random = new Random(seed);
            return Tensor.Random(new long[] { batchSize, inputSize }, random);
        }

        /// <summary>
        /// Run forward pass through model
        /// </summary>
        public static Tensor ForwardPass(Module model, Tensor input)
        {
            return model.Forward(input);
        }

        /// <summary>
        /// Run backward pass and collect gradients
        /// </summary>
        public static Dictionary<string, Tensor> BackwardPass(
            Module model,
            Tensor output,
            Tensor gradOutput)
        {
            var gradInput = model.Backward(gradOutput);
            var grads = new Dictionary<string, Tensor>();

            CollectGradients(model, grads, "");
            return grads;
        }

        private static void CollectGradients(
            Module module,
            Dictionary<string, Tensor> grads,
            string prefix)
        {
            foreach (var param in module.Parameters)
            {
                if (param.Grad != null)
                {
                    string fullName = string.IsNullOrEmpty(prefix) ? param.Name : $"{prefix}.{param.Name}";
                    grads[fullName] = param.Grad;
                }
            }

            foreach (var submodule in module.Modules)
            {
                string newPrefix = string.IsNullOrEmpty(prefix)
                    ? submodule.Name
                    : $"{prefix}.{submodule.Name}";
                CollectGradients(submodule, grads, newPrefix);
            }
        }

        /// <summary>
        /// Compare two tensors for approximate equality
        /// </summary>
        public static bool TensorsApproxEqual(Tensor a, Tensor b, double tolerance = 1e-5)
        {
            var diff = (a - b).Abs();
            var maxDiff = diff.Max().ToScalar();
            return maxDiff < tolerance;
        }

        /// <summary>
        /// Run training step
        /// </summary>
        public static void TrainingStep(
            Module model,
            Tensor input,
            Tensor target,
            Optimizer optimizer,
            Loss loss)
        {
            optimizer.ZeroGrad();

            var output = model.Forward(input);
            var lossValue = loss.Compute(output, target);

            // Backward
            var gradLoss = lossValue.Backward();
            model.Backward(gradLoss);

            // Optimizer step
            optimizer.Step();
        }
    }

    /// <summary>
    /// Mock Tensor Parallel Context for testing
    /// </summary>
    public class MockTensorParallelContext : IDisposable
    {
        public int WorldSize { get; }
        public int Rank { get; }
        private bool _disposed = false;

        public MockTensorParallelContext(int worldSize, int rank)
        {
            WorldSize = worldSize;
            Rank = rank;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Mock Column Parallel Linear layer for testing
    /// </summary>
    public class MockColumnParallelLinear : Module
    {
        private readonly LinearLayer _linear;
        private readonly bool _gatherOutput;
        private readonly MockTensorParallelContext _context;

        public MockColumnParallelLinear(
            int inputSize,
            int outputSize,
            bool bias = true,
            bool gatherOutput = false,
            MockTensorParallelContext context = null)
        {
            _gatherOutput = gatherOutput;
            _context = context;

            int localOutputSize = context != null ? outputSize / context.WorldSize : outputSize;
            _linear = new LinearLayer(inputSize, localOutputSize, bias: bias);
        }

        public Tensor GetLocalWeight()
        {
            return _linear.Weight;
        }

        public override Tensor Forward(Tensor input)
        {
            return _linear.Forward(input);
        }
    }

    /// <summary>
    /// Mock Row Parallel Linear layer for testing
    /// </summary>
    public class MockRowParallelLinear : Module
    {
        private readonly LinearLayer _linear;
        private readonly bool _inputIsSharded;
        private readonly MockTensorParallelContext _context;

        public MockRowParallelLinear(
            int inputSize,
            int outputSize,
            bool bias = true,
            bool inputIsSharded = true,
            MockTensorParallelContext context = null)
        {
            _inputIsSharded = inputIsSharded;
            _context = context;

            int localInputSize = context != null && inputIsSharded ? inputSize / context.WorldSize : inputSize;
            _linear = new LinearLayer(localInputSize, outputSize, bias: bias);
        }

        public override Tensor Forward(Tensor input)
        {
            return _linear.Forward(input);
        }
    }

    /// <summary>
    /// Mock TP Checkpoint Manager for testing
    /// </summary>
    public class TPCheckpointManager
    {
        public static System.Threading.Tasks.Task SaveDistributedAsync(Module model, string checkpointDir)
        {
            return System.Threading.Tasks.Task.Run(() =>
            {
                // Simplified: just create a marker file
                System.IO.Directory.CreateDirectory(checkpointDir);
                System.IO.File.WriteAllText(
                    System.IO.Path.Combine(checkpointDir, "checkpoint.json"),
                    "{ \"model\": \"mock\" }");
            });
        }

        public static System.Threading.Tasks.Task LoadDistributedAsync(Module model, string checkpointDir)
        {
            return System.Threading.Tasks.Task.CompletedTask;
        }

        public static List<string> ListCheckpoints(string rootDir)
        {
            if (!System.IO.Directory.Exists(rootDir))
                return new List<string>();

            return System.IO.Directory.GetDirectories(rootDir)
                .Select(d => System.IO.Path.GetFileName(d))
                .Where(d => System.IO.File.Exists(System.IO.Path.Combine(rootDir, d, "checkpoint.json")))
                .ToList();
        }
    }

    /// <summary>
    /// Mock Device Mesh for testing
    /// </summary>
    public class DeviceMesh
    {
        public MeshCoordinate MyCoordinate { get; }
        private readonly ProcessGroup _tpGroup;
        private readonly ProcessGroup _dpGroup;

        public DeviceMesh(MeshCoordinate coord, ProcessGroup tpGroup, ProcessGroup dpGroup)
        {
            MyCoordinate = coord;
            _tpGroup = tpGroup;
            _dpGroup = dpGroup;
        }

        public static DeviceMesh CreateFromRank(int rank, int[] meshShape, ProcessGroup processGroup)
        {
            // Convert rank to multi-dimensional coordinate
            var coord = RankToCoordinate(rank, meshShape);
            return new DeviceMesh(coord, processGroup, processGroup);
        }

        private static MeshCoordinate RankToCoordinate(int rank, int[] shape)
        {
            int[] indices = new int[shape.Length];
            int tempRank = rank;

            for (int i = 0; i < shape.Length; i++)
            {
                indices[i] = tempRank % shape[i];
                tempRank /= shape[i];
            }

            return new MeshCoordinate(indices);
        }

        public ProcessGroup GetTPGroup()
        {
            return _tpGroup;
        }

        public ProcessGroup GetDPGroup()
        {
            return _dpGroup;
        }
    }

    /// <summary>
    /// Mesh coordinate for DeviceMesh
    /// </summary>
    public class MeshCoordinate
    {
        public int[] Indices { get; }
        public int Dimensions => Indices.Length;

        public int this[int index] => Indices[index];

        public MeshCoordinate(int[] indices)
        {
            Indices = indices;
        }
    }

    /// <summary>
    /// Mock Loss function for testing
    /// </summary>
    public abstract class Loss
    {
        public abstract Tensor Compute(Tensor output, Tensor target);
    }

    /// <summary>
    /// Mock MSE Loss function for testing
    /// </summary>
    public class MSELoss : Loss
    {
        public override Tensor Compute(Tensor output, Tensor target)
        {
            var diff = output - target;
            var squared = diff.Mul(diff);
            return squared.Mean();
        }
    }

    /// <summary>
    /// Mock Optimizer for testing
    /// </summary>
    public class Optimizer
    {
        private readonly IEnumerable<Parameter> _parameters;
        private readonly float _learningRate;

        public Optimizer(IEnumerable<Parameter> parameters, float learningRate = 0.01f)
        {
            _parameters = parameters;
            _learningRate = learningRate;
        }

        public void ZeroGrad()
        {
            foreach (var param in _parameters)
            {
                if (param.Grad != null)
                {
                    var zeros = Tensor.ZerosLike(param.Grad);
                    param.Grad.Data = zeros.Data;
                }
            }
        }

        public void Step()
        {
            foreach (var param in _parameters)
            {
                if (param.Grad != null && param.RequiresGrad)
                {
                    var grad = param.Grad.Mul(_learningRate);
                    param.Data.Sub_(grad);
                }
            }
        }

        public float LearningRate => _learningRate;
    }

    /// <summary>
    /// Mock Adam Optimizer for testing
    /// </summary>
    public class Adam : Optimizer
    {
        public Adam(IEnumerable<Parameter> parameters, float learningRate = 0.01f)
            : base(parameters, learningRate)
        {
            // Simplified Adam implementation for testing
        }
    }
}
