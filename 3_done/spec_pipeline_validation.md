# Spec: Pipeline Validation

## Overview
Implement validation and consistency checking for pipeline parallelism, including gradient correctness verification and error handling.

## Class Design

### PipelineValidator
```csharp
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
            IPipelineCommunicator communicator);

        /// <summary>
        /// Validate pipeline configuration
        /// </summary>
        /// <returns>True if valid, false otherwise</returns>
        public bool ValidateConfiguration();

        /// <summary>
        /// Validate that all stages have consistent parameters
        /// </summary>
        public bool ValidateParameterConsistency();

        /// <summary>
        /// Validate gradient correctness by comparing with single-device baseline
        /// </summary>
        public bool ValidateGradients(Module singleDeviceModel, Tensor input);

        /// <summary>
        /// Validate numerical stability of activations and gradients
        /// </summary>
        public bool ValidateNumericalStability(List<Tensor> activations, List<Tensor> gradients);

        /// <summary>
        /// Validate memory usage is within limits
        /// </summary>
        public bool ValidateMemoryUsage(long maxMemoryBytes);

        /// <summary>
        /// Validate communication is working correctly
        /// </summary>
        public async Task<bool> ValidateCommunicationAsync();

        /// <summary>
        /// Run all validation checks
        /// </summary>
        public bool ValidateAll();

        /// <summary>
        /// Clear all errors and warnings
        /// </summary>
        public void Clear();

        public void Dispose();
    }
}
```

### ValidationError
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Represents a validation error
    /// </summary>
    public class ValidationError
    {
        /// <summary>
        /// Error code
        /// </summary>
        public string Code { get; }

        /// <summary>
        /// Error message
        /// </summary>
        public string Message { get; }

        /// <summary>
        /// Stage index where error occurred (-1 if not stage-specific)
        /// </summary>
        public int StageIndex { get; }

        /// <summary>
        /// Severity (Error = must fix)
        /// </summary>
        public ValidationSeverity Severity { get; }

        /// <summary>
        /// Additional context
        /// </summary>
        public Dictionary<string, object> Context { get; }

        public ValidationError(
            string code,
            string message,
            int stageIndex = -1,
            Dictionary<string, object>? context = null);
    }

    public enum ValidationSeverity
    {
        Error,      // Must fix before proceeding
        Warning,    // Should fix but can proceed
        Info        // Informational
    }

    public class ValidationWarning : ValidationError
    {
        public ValidationWarning(
            string code,
            string message,
            int stageIndex = -1,
            Dictionary<string, object>? context = null)
            : base(code, message, stageIndex, ValidationSeverity.Warning, context)
        {
        }
    }
}
```

### ValidationMetrics
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Metrics from pipeline validation
    /// </summary>
    public class ValidationMetrics
    {
        /// <summary>
        /// Gradient L2 norm difference (pipeline vs single-device)
        /// </summary>
        public float GradientDifference { get; }

        /// <summary>
        /// Parameter L2 norm difference (after sync)
        /// </summary>
        public float ParameterDifference { get; }

        /// <summary>
        /// Maximum activation value
        /// </summary>
        public float MaxActivation { get; }

        /// <summary>
        /// Minimum activation value
        /// </summary>
        public float MinActivation { get; }

        /// <summary>
        /// Maximum gradient value
        /// </summary>
        public float MaxGradient { get; }

        /// <summary>
        /// Minimum gradient value
        /// </summary>
        public float MinGradient { get; }

        /// <summary>
        /// Number of NaN/Inf values in activations
        /// </summary>
        public int NaNInfCount { get; }

        /// <summary>
        /// Memory usage per stage (in bytes)
        /// </summary>
        public long[] MemoryUsage { get; }

        public ValidationMetrics(
            float gradientDifference,
            float parameterDifference,
            float maxActivation,
            float minActivation,
            float maxGradient,
            float minGradient,
            int nanInfCount,
            long[] memoryUsage);
    }
}
```

## Implementation Requirements

### ValidateConfiguration
1. Check that stages list is not empty
2. Check that communicator world size matches stages count
3. Check that each stage has a valid rank
4. Check that stages are ordered correctly (0 to N-1)
5. Check that all stages are on valid devices
6. Add errors/warnings for each issue found

### ValidateParameterConsistency
1. Compare parameters across all stages
2. For models with full replicas: check that parameters match
3. For partitioned models: check that parameters are unique and cover all layers
4. Check parameter shapes match
5. Add errors if inconsistencies found

### ValidateGradients
1. Run forward+backward on single-device model
2. Run forward+backward on pipeline
3. Compare gradients (L2 norm difference)
4. Check that difference is within tolerance (e.g., 1e-4)
5. Add warning if difference is high but acceptable
6. Add error if difference is too high

### ValidateNumericalStability
1. Check for NaN and Inf values in activations
2. Check for NaN and Inf values in gradients
3. Check for very large/small values (potential overflow/underflow)
4. Report statistics (min, max, mean, std)
5. Add errors/warnings for numerical issues

### ValidateMemoryUsage
1. Estimate memory usage for each stage
2. Check if any stage exceeds maxMemoryBytes
3. Add warning if memory is close to limit (90%)
4. Add error if memory exceeds limit

### ValidateCommunicationAsync
1. Send test tensor between all adjacent stages
2. Verify received tensor matches sent tensor
3. Test bidirectional communication
4. Measure communication latency
5. Add error if communication fails or data corrupted

### Error Codes
Define standard error codes:
- `CONFIG_EMPTY_STAGES` - Stages list is empty
- `CONFIG_RANK_MISMATCH` - Stage ranks are invalid
- `CONFIG_DEVICE_MISMATCH` - Invalid device
- `PARAM_SHAPE_MISMATCH` - Parameter shapes don't match
- `PARAM_VALUE_MISMATCH` - Parameter values don't match
- `GRADIENT_MISMATCH` - Pipeline gradients differ from baseline
- `NUMERICAL_NAN` - NaN values detected
- `NUMERICAL_INF` - Inf values detected
- `COMMUNICATION_FAILURE` - Communication test failed
- `MEMORY_EXCEEDED` - Memory usage exceeds limit

## Testing Requirements

1. **Unit Tests**
   - Test configuration validation
   - Test parameter consistency validation
   - Test numerical stability validation (detect NaN/Inf)
   - Test memory usage validation
   - Test error and warning collection
   - Test error code generation
   - Test validation metrics calculation

2. **Integration Tests**
   - Test full validation with actual pipeline
   - Test gradient validation against single-device baseline
   - Test communication validation between stages
   - Test validation with intentional errors (ensure they're caught)

3. **Edge Cases**
   - Test with single-stage pipeline
   - Test with inconsistent parameters
   - Test with NaN/Inf values
   - Test with very large memory usage
   - Test with communication failures

## Files to Create
- `src/Pipeline/PipelineValidator.cs`
- `src/Pipeline/ValidationError.cs`
- `src/Pipeline/ValidationMetrics.cs`
- `tests/Pipeline/PipelineValidatorTests.cs`

## Dependencies
- `PipelineStage` from spec_pipeline_stage_core
- `IPipelineCommunicator` from spec_pipeline_communication
- Existing `Module`, `Tensor` classes
- No new external dependencies

## Time Estimate
30-45 minutes for implementation and tests

## Notes
- Validation is critical for debugging pipeline issues
- Run validation before training starts
- Run validation periodically during long training runs
- Provide clear error messages to help users diagnose issues
- Consider adding automatic recovery for some errors
