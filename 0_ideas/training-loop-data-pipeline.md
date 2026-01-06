# Training Loop with Data Pipeline

## Overview
Create a high-level training orchestration system with automatic data batching, epoch management, metrics tracking, and checkpoint/resume capabilities.

## Problem
Users currently need to manually implement training loops, handle data batching, track metrics, and manage state checkpoints. This leads to duplicated code and training inconsistencies.

## Feature Requirements

### Dataset Interface
```csharp
public interface IDataset
{
    int Length { get; }
    (Tensor Features, Tensor Labels) GetItem(int index);
}

public interface ITransform<T>
{
    T Apply(T input);
}
```

### DataLoader System
- Batched data loading with configurable batch size
- Shuffling with reproducible random seed
- Drop-last batch option for consistent batch sizes
- Parallel data loading with multiple workers
- Custom collate functions for complex data types

### Training Loop Manager
```csharp
public class Trainer
{
    public void Train(Module model, DataLoader trainLoader, DataLoader valLoader);
    public void Validate(Module model, DataLoader valLoader);
    public event EventHandler<TrainingEvent> OnEpochEnd;
    public event EventHandler<TrainingEvent> OnBatchEnd;
}
```

### Built-in Metrics
- Loss tracking (running and epoch-level)
- Accuracy, precision, recall, F1-score
- Confusion matrix computation
- Top-K accuracy for classification
- Custom metric registration

### Checkpoint Management
- Automatic model state saving at configurable intervals
- Best model checkpointing based on validation metric
- Checkpoint resume functionality (restore training state)
- Configurable checkpoint directory and naming scheme

### Training Event System
- Events for epoch start, batch start, batch end, epoch end
- Access to current epoch, batch index, metrics, gradients
- Custom event handlers for logging, early stopping, etc.

### Early Stopping
- Configurable patience (epochs without improvement)
- Min/max delta for meaningful improvement
- Best model restoration on stopping

### Usage Example
```csharp
var trainDataset = new MyTrainingDataset();
var valDataset = new MyValidationDataset();

var trainLoader = new DataLoader(trainDataset, batchSize: 32, shuffle: true);
var valLoader = new DataLoader(valDataset, batchSize: 32, shuffle: false);

var trainer = new Trainer
{
    MaxEpochs = 100,
    CheckpointInterval = 5,
    EarlyStoppingPatience = 10
};

trainer.OnBatchEnd += (sender, e) =>
{
    Console.WriteLine($"Batch {e.BatchIndex}: Loss = {e.Loss:F4}");
};

trainer.OnEpochEnd += (sender, e) =>
{
    Console.WriteLine($"Epoch {e.Epoch}: Val Loss = {e.ValidationLoss:F4}, Acc = {e.Accuracy:F2}%");
};

trainer.Train(model, trainLoader, valLoader);
```

## Technical Considerations
- Memory-efficient data loading (avoid loading entire dataset)
- Thread-safe metric accumulation
- Checkpoint compatibility across framework versions
- Integration with TensorBoard-style logging

## Value
- Reduces training boilerplate by 90%
- Provides consistent, production-ready training workflows
- Enables easy experimentation with different training configurations
- Supports complex training scenarios (multi-GPU, distributed training foundation)
