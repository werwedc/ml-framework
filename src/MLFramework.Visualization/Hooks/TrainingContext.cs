using System;
using System.Collections.Generic;

namespace MLFramework.Visualization.Hooks;

/// <summary>
/// Context object that contains information about the current training state.
/// This object is passed to hooks to provide them with context about the training process.
/// </summary>
public class TrainingContext
{
    /// <summary>
    /// Current training step (total number of batches processed so far)
    /// </summary>
    public long CurrentStep { get; set; }

    /// <summary>
    /// Current epoch number (0-based)
    /// </summary>
    public int CurrentEpoch { get; set; }

    /// <summary>
    /// Current batch number within the current epoch (0-based)
    /// </summary>
    public int CurrentBatch { get; set; }

    /// <summary>
    /// Total number of batches in the current epoch
    /// </summary>
    public int TotalBatches { get; set; }

    /// <summary>
    /// Total number of epochs to train for
    /// </summary>
    public int TotalEpochs { get; set; }

    /// <summary>
    /// Current loss value
    /// </summary>
    public float Loss { get; set; }

    /// <summary>
    /// Dictionary of metrics collected during training (e.g., accuracy, precision, recall)
    /// </summary>
    public Dictionary<string, float> Metrics { get; set; } = new Dictionary<string, float>();

    /// <summary>
    /// Current learning rate
    /// </summary>
    public float LearningRate { get; set; }

    /// <summary>
    /// Current training phase: "train", "validation", or "test"
    /// </summary>
    public string Phase { get; set; } = "train";

    /// <summary>
    /// Gets the percentage of epochs completed
    /// </summary>
    public float EpochProgress => TotalEpochs > 0 ? (float)CurrentEpoch / TotalEpochs : 0f;

    /// <summary>
    /// Gets the percentage of batches completed in the current epoch
    /// </summary>
    public float BatchProgress => TotalBatches > 0 ? (float)CurrentBatch / TotalBatches : 0f;

    /// <summary>
    /// Creates a copy of this training context
    /// </summary>
    public TrainingContext Clone()
    {
        return new TrainingContext
        {
            CurrentStep = CurrentStep,
            CurrentEpoch = CurrentEpoch,
            CurrentBatch = CurrentBatch,
            TotalBatches = TotalBatches,
            TotalEpochs = TotalEpochs,
            Loss = Loss,
            Metrics = new Dictionary<string, float>(Metrics),
            LearningRate = LearningRate,
            Phase = Phase
        };
    }

    /// <summary>
    /// Validates the context values and throws if invalid
    /// </summary>
    public void Validate()
    {
        if (CurrentStep < 0)
            throw new InvalidOperationException("CurrentStep must be non-negative");

        if (CurrentEpoch < 0)
            throw new InvalidOperationException("CurrentEpoch must be non-negative");

        if (CurrentBatch < 0)
            throw new InvalidOperationException("CurrentBatch must be non-negative");

        if (TotalBatches < 0)
            throw new InvalidOperationException("TotalBatches must be non-negative");

        if (TotalEpochs < 0)
            throw new InvalidOperationException("TotalEpochs must be non-negative");

        if (CurrentEpoch >= TotalEpochs)
            throw new InvalidOperationException("CurrentEpoch must be less than TotalEpochs");

        if (CurrentBatch >= TotalBatches)
            throw new InvalidOperationException("CurrentBatch must be less than TotalBatches");

        if (Phase != "train" && Phase != "validation" && Phase != "test")
            throw new InvalidOperationException("Phase must be 'train', 'validation', or 'test'");
    }
}
