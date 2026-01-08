namespace MachineLearning.Checkpointing;

/// <summary>
/// Types of optimizers supported by the framework
/// </summary>
public enum OptimizerType
{
    /// <summary>Stochastic Gradient Descent</summary>
    SGD,
    /// <summary>Adam optimizer</summary>
    Adam,
    /// <summary>AdamW optimizer with decoupled weight decay</summary>
    AdamW,
    /// <summary>RMSprop optimizer</summary>
    RMSprop,
    /// <summary>Adagrad optimizer</summary>
    Adagrad,
    /// <summary>Adadelta optimizer</summary>
    Adadelta,
    /// <summary>Nesterov-accelerated SGD</summary>
    Nesterov,
    /// <summary>Lion optimizer</summary>
    Lion,
    /// <summary>Custom optimizer</summary>
    Custom
}
