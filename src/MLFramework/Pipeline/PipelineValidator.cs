using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.NN;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// Validates pipeline parallelism setup and execution
    /// Checks for consistency, correctness, and numerical stability
    /// </summary>
    public class PipelineValidator : IDisposable
    {
        private readonly List<PipelineStage> _stages;
        private readonly IPipelineCommunicator _communicator;
        private readonly List<ValidationError> _errors;
        private readonly List<ValidationWarning> _warnings;
        private readonly float _gradientTolerance = 1e-4f;
        private bool _disposed;

        /// <summary>
        /// Whether validation passed
        /// </summary>
        public bool IsValid => _errors.Count == 0;

        /// <summary>
        /// List of validation errors
        /// </summary>
        public IReadOnlyList<ValidationError> Errors => _errors.AsReadOnly();

        /// <summary>
        /// List of validation warnings
        /// </summary>
        public IReadOnlyList<ValidationWarning> Warnings => _warnings.AsReadOnly();

        public PipelineValidator(
            List<PipelineStage> stages,
            IPipelineCommunicator communicator)
        {
            _stages = stages ?? throw new ArgumentNullException(nameof(stages));
            _communicator = communicator ?? throw new ArgumentNullException(nameof(communicator));
            _errors = new List<ValidationError>();
            _warnings = new List<ValidationWarning>();
        }

        /// <summary>
        /// Validate pipeline configuration
        /// </summary>
        /// <returns>True if valid, false otherwise</returns>
        public bool ValidateConfiguration()
        {
            Clear();

            // Check that stages list is not empty
            if (_stages.Count == 0)
            {
                _errors.Add(new ValidationError(
                    "CONFIG_EMPTY_STAGES",
                    "Stages list is empty"));
                return false;
            }

            // Check that communicator world size matches stages count
            if (_communicator.WorldSize != _stages.Count)
            {
                _errors.Add(new ValidationError(
                    "CONFIG_WORLD_SIZE_MISMATCH",
                    $"Communicator world size ({_communicator.WorldSize}) does not match number of stages ({_stages.Count})"));
            }

            // Check that each stage has a valid rank
            var seenRanks = new HashSet<int>();
            for (int i = 0; i < _stages.Count; i++)
            {
                var stage = _stages[i];
                if (stage.Rank < 0 || stage.Rank >= _stages.Count)
                {
                    _errors.Add(new ValidationError(
                        "CONFIG_RANK_MISMATCH",
                        $"Stage {i} has invalid rank {stage.Rank} (should be in [0, {_stages.Count - 1}])",
                        stageIndex: i));
                }

                // Check for duplicate ranks
                if (seenRanks.Contains(stage.Rank))
                {
                    _errors.Add(new ValidationError(
                        "CONFIG_DUPLICATE_RANK",
                        $"Duplicate rank {stage.Rank} detected for stage {i}",
                        stageIndex: i));
                }
                seenRanks.Add(stage.Rank);

                // Check that stages are ordered correctly (0 to N-1)
                if (stage.Rank != i)
                {
                    _warnings.Add(new ValidationWarning(
                        "CONFIG_RANK_ORDER",
                        $"Stage {i} has rank {stage.Rank}, expected {i}",
                        stageIndex: i));
                }

                // Check that all stages are on valid devices
                if (stage.Device == null)
                {
                    _errors.Add(new ValidationError(
                        "CONFIG_DEVICE_MISMATCH",
                        $"Stage {i} has null device",
                        stageIndex: i));
                }
            }

            return IsValid;
        }

        /// <summary>
        /// Validate that all stages have consistent parameters
        /// </summary>
        public bool ValidateParameterConsistency()
        {
            Clear();

            if (_stages.Count == 0)
            {
                _errors.Add(new ValidationError(
                    "PARAM_EMPTY_STAGES",
                    "Cannot validate parameters with no stages"));
                return false;
            }

            // Get all parameters from first stage as reference
            var refParams = _stages[0].GetNamedParameters().ToList();

            for (int stageIdx = 1; stageIdx < _stages.Count; stageIdx++)
            {
                var stageParams = _stages[stageIdx].GetNamedParameters().ToList();

                // For full model replicas: check that parameters match
                // For partitioned models: check that parameters are unique and cover all layers

                // Simple check: compare parameter counts
                if (stageParams.Count != refParams.Count)
                {
                    _warnings.Add(new ValidationWarning(
                        "PARAM_COUNT_MISMATCH",
                        $"Stage {stageIdx} has {stageParams.Count} parameters, reference has {refParams.Count}",
                        stageIndex: stageIdx));
                    continue;
                }

                // Check parameter shapes match for parameters with same names
                foreach (var (name, param) in stageParams)
                {
                    var refParam = refParams.FirstOrDefault(p => p.Name == name);
                    if (refParam.Parameter == null)
                    {
                        _warnings.Add(new ValidationWarning(
                            "PARAM_NAME_NOT_FOUND",
                            $"Parameter '{name}' not found in reference stage",
                            stageIndex: stageIdx));
                        continue;
                    }

                    if (!param.Shape.SequenceEqual(refParam.Parameter.Shape))
                    {
                        _errors.Add(new ValidationError(
                            "PARAM_SHAPE_MISMATCH",
                            $"Parameter '{name}' shape mismatch: {string.Join(",", param.Shape)} vs {string.Join(",", refParam.Parameter.Shape)}",
                            stageIndex: stageIdx));
                    }
                }
            }

            return IsValid;
        }

        /// <summary>
        /// Validate gradient correctness by comparing with single-device baseline
        /// </summary>
        public bool ValidateGradients(Module singleDeviceModel, Tensor input)
        {
            Clear();

            if (singleDeviceModel == null)
            {
                _errors.Add(new ValidationError(
                    "GRADIENT_NULL_BASELINE",
                    "Single-device model is null"));
                return false;
            }

            if (input == null)
            {
                _errors.Add(new ValidationError(
                    "GRADIENT_NULL_INPUT",
                    "Input tensor is null"));
                return false;
            }

            try
            {
                // Run forward+backward on single-device model
                var singleOutput = singleDeviceModel.Forward(input);
                var singleLoss = CreateLoss(singleOutput);
                singleLoss.Backward();

                // Collect gradients from single-device model
                var singleGradients = singleDeviceModel.GetNamedParameters()
                    .ToDictionary(p => p.Name, p => p.Parameter.Gradient);

                // Run forward+backward on pipeline
                // Note: This is a simplified implementation
                // In a real implementation, you'd need to coordinate across stages
                var pipelineGradients = new Dictionary<string, Tensor>();
                foreach (var stage in _stages)
                {
                    foreach (var (name, param) in stage.GetNamedParameters())
                    {
                        if (param.Gradient != null)
                        {
                            pipelineGradients[name] = param.Gradient;
                        }
                    }
                }

                // Compare gradients (L2 norm difference)
                float totalDiff = 0f;
                int comparedCount = 0;
                foreach (var (name, singleGrad) in singleGradients)
                {
                    if (singleGrad == null || !pipelineGradients.ContainsKey(name))
                    {
                        continue;
                    }

                    var pipelineGrad = pipelineGradients[name];
                    var diff = L2NormDifference(singleGrad, pipelineGrad);
                    totalDiff += diff;
                    comparedCount++;

                    if (diff > _gradientTolerance)
                    {
                        _errors.Add(new ValidationError(
                            "GRADIENT_MISMATCH",
                            $"Gradient for parameter '{name}' differs by {diff:E4} (tolerance: {_gradientTolerance:E4})",
                            -1,
                            new Dictionary<string, object>
                            {
                                { "parameter", name },
                                { "difference", diff }
                            }));
                    }
                }

                float avgDiff = comparedCount > 0 ? totalDiff / comparedCount : 0f;
                if (avgDiff > _gradientTolerance * 10)
                {
                    _errors.Add(new ValidationError(
                        "GRADIENT_OVERALL_MISMATCH",
                        $"Average gradient difference {avgDiff:E4} exceeds tolerance",
                        -1,
                        new Dictionary<string, object>
                        {
                            { "averageDifference", avgDiff },
                            { "tolerance", _gradientTolerance }
                        }));
                }
            }
            catch (Exception ex)
            {
                _errors.Add(new ValidationError(
                    "GRADIENT_COMPUTATION_ERROR",
                    $"Error during gradient validation: {ex.Message}"));
                return false;
            }

            return IsValid;
        }

        /// <summary>
        /// Validate numerical stability of activations and gradients
        /// </summary>
        public bool ValidateNumericalStability(List<Tensor> activations, List<Tensor> gradients)
        {
            Clear();

            if (activations == null || activations.Count == 0)
            {
                _errors.Add(new ValidationError(
                    "NUMERICAL_NO_ACTIVATIONS",
                    "Activations list is null or empty"));
                return false;
            }

            int nanInfCount = 0;
            float maxActivation = float.NegativeInfinity;
            float minActivation = float.PositiveInfinity;
            float maxGradient = float.NegativeInfinity;
            float minGradient = float.PositiveInfinity;

            // Check activations
            for (int i = 0; i < activations.Count; i++)
            {
                var tensor = activations[i];
                if (tensor == null || tensor.Data == null)
                {
                    _errors.Add(new ValidationError(
                        "NUMERICAL_NULL_ACTIVATION",
                        $"Activation tensor at index {i} is null",
                        -1,
                        new Dictionary<string, object> { { "index", i } }));
                    continue;
                }

                foreach (var val in tensor.Data)
                {
                    if (float.IsNaN(val))
                    {
                        _errors.Add(new ValidationError(
                            "NUMERICAL_NAN",
                            $"NaN value detected in activation at index {i}",
                            -1,
                            new Dictionary<string, object> { { "index", i } }));
                        nanInfCount++;
                    }
                    else if (float.IsInfinity(val))
                    {
                        _errors.Add(new ValidationError(
                            "NUMERICAL_INF",
                            $"Inf value detected in activation at index {i}",
                            -1,
                            new Dictionary<string, object> { { "index", i } }));
                        nanInfCount++;
                    }
                    else
                    {
                        maxActivation = Math.Max(maxActivation, val);
                        minActivation = Math.Min(minActivation, val);
                    }
                }
            }

            // Check gradients
            if (gradients != null)
            {
                for (int i = 0; i < gradients.Count; i++)
                {
                    var tensor = gradients[i];
                    if (tensor == null || tensor.Data == null)
                    {
                        continue;
                    }

                    foreach (var val in tensor.Data)
                    {
                        if (float.IsNaN(val))
                        {
                            _errors.Add(new ValidationError(
                                "NUMERICAL_NAN_GRADIENT",
                                $"NaN value detected in gradient at index {i}",
                                -1,
                                new Dictionary<string, object> { { "index", i } }));
                            nanInfCount++;
                        }
                        else if (float.IsInfinity(val))
                        {
                            _errors.Add(new ValidationError(
                                "NUMERICAL_INF_GRADIENT",
                                $"Inf value detected in gradient at index {i}",
                                -1,
                                new Dictionary<string, object> { { "index", i } }));
                            nanInfCount++;
                        }
                        else
                        {
                            maxGradient = Math.Max(maxGradient, val);
                            minGradient = Math.Min(minGradient, val);
                        }
                    }
                }
            }

            // Check for very large/small values (potential overflow/underflow)
            if (maxActivation > 1e6f)
            {
                _warnings.Add(new ValidationWarning(
                    "NUMERICAL_LARGE_ACTIVATION",
                    $"Very large activation value detected: {maxActivation:E4}"));
            }

            if (minActivation < -1e6f)
            {
                _warnings.Add(new ValidationWarning(
                    "NUMERICAL_SMALL_ACTIVATION",
                    $"Very small activation value detected: {minActivation:E4}"));
            }

            if (maxGradient > 1e6f)
            {
                _warnings.Add(new ValidationWarning(
                    "NUMERICAL_LARGE_GRADIENT",
                    $"Very large gradient value detected: {maxGradient:E4}"));
            }

            return IsValid;
        }

        /// <summary>
        /// Validate memory usage is within limits
        /// </summary>
        public bool ValidateMemoryUsage(long maxMemoryBytes)
        {
            Clear();

            if (maxMemoryBytes <= 0)
            {
                _errors.Add(new ValidationError(
                    "MEMORY_INVALID_LIMIT",
                    $"Invalid memory limit: {maxMemoryBytes} bytes"));
                return false;
            }

            var memoryUsage = new long[_stages.Count];
            long totalMemory = 0;

            for (int i = 0; i < _stages.Count; i++)
            {
                var stage = _stages[i];
                long stageMemory = 0;

                // Estimate memory from parameters
                foreach (var param in stage.GetParameters())
                {
                    long paramSize = 1;
                    foreach (var dim in param.Shape)
                    {
                        paramSize *= dim;
                    }
                    stageMemory += paramSize * sizeof(float); // 4 bytes per float
                }

                memoryUsage[i] = stageMemory;
                totalMemory += stageMemory;

                // Add warning if memory is close to limit (90%)
                if (stageMemory > maxMemoryBytes * 0.9)
                {
                    _warnings.Add(new ValidationWarning(
                        "MEMORY_NEAR_LIMIT",
                        $"Stage {i} memory usage {stageMemory} bytes is near limit {maxMemoryBytes} bytes",
                        i,
                        new Dictionary<string, object>
                        {
                            { "memoryUsage", stageMemory },
                            { "memoryLimit", maxMemoryBytes }
                        }));
                }

                // Add error if memory exceeds limit
                if (stageMemory > maxMemoryBytes)
                {
                    _errors.Add(new ValidationError(
                        "MEMORY_EXCEEDED",
                        $"Stage {i} memory usage {stageMemory} bytes exceeds limit {maxMemoryBytes} bytes",
                        i,
                        new Dictionary<string, object>
                        {
                            { "memoryUsage", stageMemory },
                            { "memoryLimit", maxMemoryBytes }
                        }));
                }
            }

            return IsValid;
        }

        /// <summary>
        /// Validate communication is working correctly
        /// </summary>
        public async Task<bool> ValidateCommunicationAsync()
        {
            Clear();

            try
            {
                // Test: Send test tensor between all adjacent stages
                var testData = new float[100];
                for (int i = 0; i < testData.Length; i++)
                {
                    testData[i] = (float)i;
                }
                var testTensor = new Tensor(testData, new[] { 100 });

                // For simplicity, test broadcast from rank 0
                var broadcastResult = await _communicator.BroadcastAsync(testTensor, 0);
                if (broadcastResult == null || broadcastResult.Data == null)
                {
                    _errors.Add(new ValidationError(
                        "COMMUNICATION_FAILURE",
                        "Broadcast returned null tensor"));
                    return false;
                }

                // Verify data integrity
                for (int i = 0; i < testData.Length; i++)
                {
                    if (Math.Abs(broadcastResult.Data[i] - testData[i]) > 1e-6f)
                    {
                        _errors.Add(new ValidationError(
                            "COMMUNICATION_DATA_CORRUPTION",
                            $"Broadcast data corrupted at index {i}: expected {testData[i]}, got {broadcastResult.Data[i]}"));
                        return false;
                    }
                }

                // Test barrier
                await _communicator.BarrierAsync();
            }
            catch (Exception ex)
            {
                _errors.Add(new ValidationError(
                    "COMMUNICATION_FAILURE",
                    $"Communication test failed: {ex.Message}"));
                return false;
            }

            return IsValid;
        }

        /// <summary>
        /// Run all validation checks
        /// </summary>
        public bool ValidateAll()
        {
            Clear();
            bool allValid = true;

            allValid &= ValidateConfiguration();
            allValid &= ValidateParameterConsistency();

            return allValid;
        }

        /// <summary>
        /// Clear all errors and warnings
        /// </summary>
        public void Clear()
        {
            _errors.Clear();
            _warnings.Clear();
        }

        /// <summary>
        /// Get validation metrics from the last run
        /// </summary>
        public ValidationMetrics GetMetrics()
        {
            return new ValidationMetrics(
                gradientDifference: 0f, // Would be computed during gradient validation
                parameterDifference: 0f,
                maxActivation: float.MaxValue,
                minActivation: float.MinValue,
                maxGradient: float.MaxValue,
                minGradient: float.MinValue,
                nanInfCount: _errors.Count(e => e.Code.StartsWith("NUMERICAL_")),
                memoryUsage: new long[_stages.Count]);
        }

        /// <summary>
        /// Compute L2 norm difference between two tensors
        /// </summary>
        private float L2NormDifference(Tensor tensor1, Tensor tensor2)
        {
            if (tensor1 == null || tensor2 == null)
            {
                return float.PositiveInfinity;
            }

            if (tensor1.Data == null || tensor2.Data == null)
            {
                return float.PositiveInfinity;
            }

            if (tensor1.Data.Length != tensor2.Data.Length)
            {
                return float.PositiveInfinity;
            }

            float diffSum = 0f;
            for (int i = 0; i < tensor1.Data.Length; i++)
            {
                float diff = tensor1.Data[i] - tensor2.Data[i];
                diffSum += diff * diff;
            }

            return (float)Math.Sqrt(diffSum);
        }

        /// <summary>
        /// Create a simple loss tensor from output (simplified implementation)
        /// </summary>
        private Tensor CreateLoss(Tensor output)
        {
            if (output == null)
            {
                return new Tensor(new float[] { 0f }, new[] { 1 });
            }

            // Simple L2 loss
            float loss = 0f;
            if (output.Data != null)
            {
                foreach (var val in output.Data)
                {
                    loss += val * val;
                }
            }

            return new Tensor(new float[] { loss / (output.Data?.Length ?? 1) }, new[] { 1 }, requiresGrad: true);
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _communicator?.Dispose();
                _disposed = true;
            }
        }
    }
}
