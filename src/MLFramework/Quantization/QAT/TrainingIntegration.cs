namespace MLFramework.Quantization.QAT;

/// <summary>
/// Provides utilities and guidelines for integrating quantization-aware training
/// into existing training pipelines.
///
/// QAT is designed to be drop-in compatible with existing training infrastructure:
/// - Optimizers: No changes required. Gradients flow through fake quant nodes via STE.
/// - Training loops: No changes required. Use the same forward/backward/optimizer.step pattern.
/// - Loss functions: No changes required. Compute loss on quantized outputs as usual.
/// </summary>
public static class TrainingIntegration
{
    /// <summary>
    /// Sets the training mode for a QAT model.
    /// During training, fake quantization nodes are active and parameters are learned.
    /// During evaluation/inference, fake quantization uses fixed learned parameters.
    /// </summary>
    /// <param name="qatModel">The QAT model.</param>
    /// <param name="trainingMode">True for training mode, false for evaluation mode.</param>
    public static void SetTrainingMode(IQATModel qatModel, bool trainingMode)
    {
        if (qatModel == null)
            throw new ArgumentNullException(nameof(qatModel));

        qatModel.TrainingMode = trainingMode;
    }

    /// <summary>
    /// Prepares a QAT model for training epoch.
    /// Ensures all fake quantization nodes are in training mode.
    /// </summary>
    /// <param name="qatModel">The QAT model.</param>
    public static void PrepareForTraining(IQATModel qatModel)
    {
        SetTrainingMode(qatModel, trainingMode: true);
    }

    /// <summary>
    /// Prepares a QAT model for evaluation.
    /// Ensures all fake quantization nodes use fixed learned parameters.
    /// </summary>
    /// <param name="qatModel">The QAT model.</param>
    public static void PrepareForEvaluation(IQATModel qatModel)
    {
        SetTrainingMode(qatModel, trainingMode: false);
    }

    /// <summary>
    /// Checks if the model is in training mode.
    /// </summary>
    /// <param name="qatModel">The QAT model.</param>
    /// <returns>True if in training mode.</returns>
    public static bool IsTrainingMode(IQATModel qatModel)
    {
        if (qatModel == null)
            throw new ArgumentNullException(nameof(qatModel));

        return qatModel.TrainingMode;
    }

    /// <summary>
    /// Performs a training step with a QAT model.
    /// This is equivalent to a standard training step but with QAT awareness.
    ///
    /// Example usage:
    /// <code>
    /// // Forward pass
    /// var output = model(input);
    ///
    /// // Compute loss (no changes needed)
    /// var loss = lossFunction(output, target);
    ///
    /// // Backward pass (gradients flow through fake quant via STE)
    /// loss.Backward();
    ///
    /// // Optimizer step (no changes needed)
    /// optimizer.Step();
    /// </code>
    /// </summary>
    /// <param name="qatModel">The QAT model.</param>
    /// <param name="input">Input tensor.</param>
    /// <param name="target">Target tensor.</param>
    /// <param name="lossFunction">Loss function.</param>
    /// <param name="optimizer">Optimizer.</param>
    /// <returns>The computed loss value.</returns>
    public static float TrainingStep(
        IQATModel qatModel,
        object input,
        object target,
        object lossFunction,
        object optimizer)
    {
        // Ensure model is in training mode
        PrepareForTraining(qatModel);

        // In production, this would execute the full training step
        // This is a placeholder for documentation purposes

        return 0.0f; // Placeholder
    }

    /// <summary>
    /// Performs an evaluation step with a QAT model.
    /// </summary>
    /// <param name="qatModel">The QAT model.</param>
    /// <param name="input">Input tensor.</param>
    /// <param name="target">Target tensor.</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    /// <returns>The computed loss value or metric.</returns>
    public static float EvaluationStep(
        IQATModel qatModel,
        object input,
        object target,
        object? lossFunction = null)
    {
        // Ensure model is in evaluation mode
        PrepareForEvaluation(qatModel);

        // In production, this would execute the evaluation step
        // This is a placeholder for documentation purposes

        return 0.0f; // Placeholder
    }

    /// <summary>
    /// Gets training guidelines for QAT models.
    /// </summary>
    /// <returns>List of training guidelines.</returns>
    public static List<string> GetTrainingGuidelines()
    {
        return new List<string>
        {
            "1. Warm-up Phase: Start with lower learning rates to stabilize quantization parameters.",
            "2. Gradual Quantization: Consider gradually increasing quantization aggressiveness.",
            "3. Validation: Monitor validation accuracy closely during QAT training.",
            "4. Quantization Parameters: Observe scale and zero-point evolution.",
            "5. Learning Rate: QAT may require slightly higher learning rates than standard training.",
            "6. Batch Size: Use consistent batch sizes for stable calibration statistics.",
            "7. Freezing: Consider freezing quantization parameters after a certain epoch.",
            "8. Mixed Precision: Enable mixed precision QAT for sensitive layers if needed."
        };
    }

    /// <summary>
    /// Gets compatibility notes for QAT training.
    /// </summary>
    /// <returns>List of compatibility notes.</returns>
    public static List<string> GetCompatibilityNotes()
    {
        return new List<string>
        {
            "Optimizers: All standard optimizers (SGD, Adam, AdamW, etc.) work with QAT.",
            "Loss Functions: All loss functions work with QAT as they operate on tensor outputs.",
            "Learning Rate Schedulers: No changes required for QAT.",
            "Gradient Clipping: Works with QAT. Apply after backward pass.",
            "Batch Normalization: Batch norm layers should be fused or handled carefully.",
            "Dropout: Works with QAT. Dropout is applied before quantization.",
            "Layer Freezing: Frozen layers still participate in quantization simulation.",
            "Distributed Training: QAT is compatible with distributed training frameworks."
        };
    }

    /// <summary>
    /// Gets recommended training schedule for QAT.
    /// </summary>
    /// <param name="totalEpochs">Total number of training epochs.</param>
    /// <returns>A training schedule with QAT-specific recommendations.</returns>
    public static QATTrainingSchedule GetRecommendedSchedule(int totalEpochs)
    {
        if (totalEpochs <= 0)
            throw new ArgumentException("Total epochs must be positive", nameof(totalEpochs));

        // Standard QAT training schedule
        var warmupEpochs = Math.Max(1, totalEpochs / 10);
        var stableEpochs = totalEpochs - warmupEpochs;

        return new QATTrainingSchedule
        {
            TotalEpochs = totalEpochs,
            WarmupEpochs = warmupEpochs,
            StableEpochs = stableEpochs,
            RecommendedInitialLR = 0.001f,
            RecommendedFinalLR = 0.0001f,
            EnableGradualQuantization = true,
            QuantizationParameterFreezeEpoch = totalEpochs * 2 / 3
        };
    }

    /// <summary>
    /// Validates training setup for QAT.
    /// </summary>
    /// <param name="qatModel">The QAT model.</param>
    /// <param name="config">Validation configuration.</param>
    /// <returns>Validation result with any warnings or errors.</returns>
    public static QATTrainingValidation ValidateTrainingSetup(IQATModel qatModel, QATTrainingValidationConfig? config = null)
    {
        var result = new QATTrainingValidation();
        config ??= new QATTrainingValidationConfig();

        // Check if model has fake quantization nodes
        var fakeQuantCount = qatModel.GetFakeQuantizationNodeCount();
        if (fakeQuantCount == 0)
        {
            result.Warnings.Add("Model has no fake quantization nodes. QAT may not be properly configured.");
        }
        else
        {
            result.Info.Add($"Model has {fakeQuantCount} fake quantization nodes.");
        }

        // Check quantized layer count
        var quantizedLayerCount = qatModel.GetQuantizedLayerCount();
        result.Info.Add($"{quantizedLayerCount} layers are configured for quantization.");

        // Check training mode
        if (qatModel.TrainingMode)
        {
            result.Info.Add("Model is in training mode.");
        }
        else
        {
            result.Warnings.Add("Model is not in training mode. Set training mode before training.");
        }

        return result;
    }
}

/// <summary>
/// Training schedule recommendations for QAT.
/// </summary>
public class QATTrainingSchedule
{
    /// <summary>
    /// Gets or sets total number of epochs.
    /// </summary>
    public int TotalEpochs { get; set; }

    /// <summary>
    /// Gets or sets number of warm-up epochs.
    /// </summary>
    public int WarmupEpochs { get; set; }

    /// <summary>
    /// Gets or sets number of stable training epochs.
    /// </summary>
    public int StableEpochs { get; set; }

    /// <summary>
    /// Gets or sets recommended initial learning rate.
    /// </summary>
    public float RecommendedInitialLR { get; set; }

    /// <summary>
    /// Gets or sets recommended final learning rate.
    /// </summary>
    public float RecommendedFinalLR { get; set; }

    /// <summary>
    /// Gets or sets whether to enable gradual quantization.
    /// </summary>
    public bool EnableGradualQuantization { get; set; }

    /// <summary>
    /// Gets or sets the epoch at which to freeze quantization parameters.
    /// </summary>
    public int QuantizationParameterFreezeEpoch { get; set; }
}

/// <summary>
/// Configuration for QAT training validation.
/// </summary>
public class QATTrainingValidationConfig
{
    /// <summary>
    /// Gets or sets whether to check for fake quantization nodes.
    /// </summary>
    public bool CheckFakeQuantNodes { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to check training mode.
    /// </summary>
    public bool CheckTrainingMode { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to check quantization parameters.
    /// </summary>
    public bool CheckQuantizationParameters { get; set; } = true;
}

/// <summary>
/// Result of QAT training validation.
/// </summary>
public class QATTrainingValidation
{
    /// <summary>
    /// Gets or sets list of validation errors.
    /// </summary>
    public List<string> Errors { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets list of validation warnings.
    /// </summary>
    public List<string> Warnings { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets list of validation info messages.
    /// </summary>
    public List<string> Info { get; set; } = new List<string>();

    /// <summary>
    /// Gets whether the validation passed (no errors).
    /// </summary>
    public bool IsValid => Errors.Count == 0;
}
