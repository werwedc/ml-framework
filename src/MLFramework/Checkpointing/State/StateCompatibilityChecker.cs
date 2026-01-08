namespace MachineLearning.Checkpointing;

/// <summary>
/// Checks compatibility between state dictionaries
/// </summary>
public class StateCompatibilityChecker
{
    /// <summary>
    /// Checks compatibility between two state dictionaries
    /// </summary>
    /// <param name="sourceState">Source state dictionary</param>
    /// <param name="targetState">Target state dictionary</param>
    /// <returns>Compatibility result with any errors or warnings</returns>
    public CompatibilityResult CheckCompatibility(
        StateDict sourceState,
        StateDict targetState)
    {
        var result = new CompatibilityResult();

        // Check for null states
        if (sourceState == null && targetState == null)
        {
            return CompatibilityResult.Success();
        }

        if (sourceState == null || targetState == null)
        {
            result.AddError("One or both state dictionaries are null");
            return result;
        }

        // Check key match
        var missingKeys = StateUtils.GetMissingKeys(sourceState, targetState);
        var unexpectedKeys = StateUtils.GetUnexpectedKeys(sourceState, targetState);

        if (missingKeys.Count > 0)
        {
            result.AddWarning($"Missing keys in target: {string.Join(", ", missingKeys)}");
        }

        if (unexpectedKeys.Count > 0)
        {
            result.AddWarning($"Unexpected keys in target: {string.Join(", ", unexpectedKeys)}");
        }

        // Check shapes for common keys
        CheckShapes(sourceState, targetState, result);

        // Check data types for common keys
        CheckDataTypes(sourceState, targetState, result);

        return result;
    }

    /// <summary>
    /// Checks compatibility with strict mode (requires exact match)
    /// </summary>
    /// <param name="sourceState">Source state dictionary</param>
    /// <param name="targetState">Target state dictionary</param>
    /// <returns>Compatibility result with any errors or warnings</returns>
    public CompatibilityResult CheckCompatibilityStrict(
        StateDict sourceState,
        StateDict targetState)
    {
        var result = CheckCompatibility(sourceState, targetState);

        // Promote warnings to errors for strict mode
        if (result.Warnings.Count > 0)
        {
            foreach (var warning in result.Warnings)
            {
                result.AddError(warning);
            }
            result.Warnings.Clear();
        }

        return result;
    }

    /// <summary>
    /// Checks compatibility between optimizer state dictionaries
    /// </summary>
    /// <param name="sourceState">Source optimizer state dictionary</param>
    /// <param name="targetState">Target optimizer state dictionary</param>
    /// <returns>Compatibility result with any errors or warnings</returns>
    public CompatibilityResult CheckOptimizerCompatibility(
        OptimizerStateDict sourceState,
        OptimizerStateDict targetState)
    {
        var result = CheckCompatibility(sourceState, targetState);

        if (sourceState == null || targetState == null)
            return result;

        // Check optimizer type
        if (sourceState.OptimizerType != targetState.OptimizerType)
        {
            result.AddError(
                $"Optimizer type mismatch: expected {sourceState.OptimizerType}, " +
                $"got {targetState.OptimizerType}");
        }

        return result;
    }

    /// <summary>
    /// Checks compatibility between model state dictionaries
    /// </summary>
    /// <param name="sourceState">Source model state dictionary</param>
    /// <param name="targetState">Target model state dictionary</param>
    /// <returns>Compatibility result with any errors or warnings</returns>
    public CompatibilityResult CheckModelCompatibility(
        ModelStateDict sourceState,
        ModelStateDict targetState)
    {
        var result = CheckCompatibility(sourceState, targetState);

        if (sourceState == null || targetState == null)
            return result;

        // Check model type
        if (!string.IsNullOrEmpty(sourceState.ModelType) &&
            !string.IsNullOrEmpty(targetState.ModelType) &&
            sourceState.ModelType != targetState.ModelType)
        {
            result.AddWarning(
                $"Model type mismatch: expected {sourceState.ModelType}, " +
                $"got {targetState.ModelType}");
        }

        // Check layer count
        if (sourceState.LayerCount > 0 &&
            targetState.LayerCount > 0 &&
            sourceState.LayerCount != targetState.LayerCount)
        {
            result.AddWarning(
                $"Layer count mismatch: expected {sourceState.LayerCount}, " +
                $"got {targetState.LayerCount}");
        }

        return result;
    }

    /// <summary>
    /// Checks tensor shape compatibility
    /// </summary>
    private void CheckShapes(
        StateDict sourceState,
        StateDict targetState,
        CompatibilityResult result)
    {
        // Find common keys
        var commonKeys = sourceState.Keys.Intersect(targetState.Keys);

        foreach (var key in commonKeys)
        {
            var sourceTensor = sourceState[key];
            var targetTensor = targetState[key];

            if (!sourceTensor.Shape.SequenceEqual(targetTensor.Shape))
            {
                result.AddError(
                    $"Shape mismatch for '{key}': " +
                    $"expected {string.Join(",", sourceTensor.Shape)}, " +
                    $"got {string.Join(",", targetTensor.Shape)}");
            }
        }
    }

    /// <summary>
    /// Checks tensor data type compatibility
    /// </summary>
    private void CheckDataTypes(
        StateDict sourceState,
        StateDict targetState,
        CompatibilityResult result)
    {
        // Find common keys
        var commonKeys = sourceState.Keys.Intersect(targetState.Keys);

        foreach (var key in commonKeys)
        {
            var sourceTensor = sourceState[key];
            var targetTensor = targetState[key];

            if (sourceTensor.DataType != targetTensor.DataType)
            {
                result.AddError(
                    $"Data type mismatch for '{key}': " +
                    $"expected {sourceTensor.DataType}, " +
                    $"got {targetTensor.DataType}");
            }
        }
    }

    /// <summary>
    /// Checks if a state dict can be loaded into a model with given architecture
    /// </summary>
    /// <param name="checkpointState">State dictionary from checkpoint</param>
    /// <param name="modelState">Current model state dictionary</param>
    /// <param name="allowPartialLoad">Whether to allow loading partial checkpoints</param>
    /// <returns>Compatibility result</returns>
    public CompatibilityResult CheckLoadCompatibility(
        StateDict checkpointState,
        StateDict modelState,
        bool allowPartialLoad = false)
    {
        var result = new CompatibilityResult();

        if (checkpointState == null || modelState == null)
        {
            result.AddError("Cannot check compatibility with null state dictionaries");
            return result;
        }

        var missingKeys = StateUtils.GetMissingKeys(checkpointState, modelState);
        var unexpectedKeys = StateUtils.GetUnexpectedKeys(checkpointState, modelState);

        // Check if checkpoint has all required keys
        if (missingKeys.Count > 0)
        {
            var errorMsg = $"Checkpoint is missing {missingKeys.Count} required keys";
            if (!allowPartialLoad)
            {
                result.AddError($"{errorMsg}: {string.Join(", ", missingKeys.Take(5))}");
                if (missingKeys.Count > 5)
                {
                    result.AddError($"... and {missingKeys.Count - 5} more keys");
                }
            }
            else
            {
                result.AddWarning($"{errorMsg}: {string.Join(", ", missingKeys)}");
            }
        }

        // Check for extra keys in checkpoint
        if (unexpectedKeys.Count > 0)
        {
            result.AddWarning(
                $"Checkpoint has {unexpectedKeys.Count} keys not present in model: " +
                $"{string.Join(", ", unexpectedKeys.Take(5))}");
            if (unexpectedKeys.Count > 5)
            {
                result.AddWarning($"... and {unexpectedKeys.Count - 5} more keys");
            }
        }

        // Check shapes for common keys
        CheckShapes(checkpointState, modelState, result);

        return result;
    }
}
