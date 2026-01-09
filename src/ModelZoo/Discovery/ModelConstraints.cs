using System;

namespace MLFramework.ModelZoo.Discovery
{
    /// <summary>
    /// Defines constraints for model recommendation.
    /// </summary>
    public class ModelConstraints
    {
        /// <summary>
        /// Gets or sets the input tensor shape.
        /// </summary>
        public Shape InputShape { get; set; }

        /// <summary>
        /// Gets or sets the target task type.
        /// </summary>
        public TaskType Task { get; set; }

        /// <summary>
        /// Gets or sets the maximum inference latency in milliseconds.
        /// Null means no constraint.
        /// </summary>
        public float? MaxLatency { get; set; }

        /// <summary>
        /// Gets or sets the maximum memory usage in bytes.
        /// Null means no constraint.
        /// </summary>
        public long? MaxMemory { get; set; }

        /// <summary>
        /// Gets or sets the minimum required accuracy (0.0 to 1.0).
        /// Null means no constraint.
        /// </summary>
        public float? MinAccuracy { get; set; }

        /// <summary>
        /// Gets or sets the maximum model file size in bytes.
        /// Null means no constraint.
        /// </summary>
        public long? MaxFileSize { get; set; }

        /// <summary>
        /// Gets or sets the target device type.
        /// Null means no constraint.
        /// </summary>
        public DeviceType? Device { get; set; }

        /// <summary>
        /// Gets or sets the expected batch size.
        /// Null means no constraint.
        /// </summary>
        public int? BatchSize { get; set; }

        /// <summary>
        /// Gets or sets the deployment environment.
        /// Null means no constraint.
        /// </summary>
        public DeploymentEnv? DeploymentEnvironment { get; set; }

        /// <summary>
        /// Initializes a new instance of the ModelConstraints class.
        /// </summary>
        public ModelConstraints()
        {
        }

        /// <summary>
        /// Initializes a new instance of the ModelConstraints class with required parameters.
        /// </summary>
        /// <param name="inputShape">The input shape.</param>
        /// <param name="task">The task type.</param>
        public ModelConstraints(Shape inputShape, TaskType task)
        {
            InputShape = inputShape ?? throw new ArgumentNullException(nameof(inputShape));
            Task = task;
        }

        /// <summary>
        /// Creates a constraint builder for fluent construction.
        /// </summary>
        /// <returns>A new ModelConstraints instance.</returns>
        public static ModelConstraints Create()
        {
            return new ModelConstraints();
        }

        /// <summary>
        /// Sets the input shape constraint.
        /// </summary>
        /// <param name="inputShape">The input shape.</param>
        /// <returns>This instance for fluent chaining.</returns>
        public ModelConstraints WithInputShape(Shape inputShape)
        {
            InputShape = inputShape ?? throw new ArgumentNullException(nameof(inputShape));
            return this;
        }

        /// <summary>
        /// Sets the task type constraint.
        /// </summary>
        /// <param name="task">The task type.</param>
        /// <returns>This instance for fluent chaining.</returns>
        public ModelConstraints WithTask(TaskType task)
        {
            Task = task;
            return this;
        }

        /// <summary>
        /// Sets the maximum latency constraint.
        /// </summary>
        /// <param name="maxLatencyMs">Maximum latency in milliseconds.</param>
        /// <returns>This instance for fluent chaining.</returns>
        public ModelConstraints WithMaxLatency(float maxLatencyMs)
        {
            if (maxLatencyMs <= 0)
                throw new ArgumentException("Max latency must be positive", nameof(maxLatencyMs));

            MaxLatency = maxLatencyMs;
            return this;
        }

        /// <summary>
        /// Sets the maximum memory constraint.
        /// </summary>
        /// <param name="maxMemoryBytes">Maximum memory in bytes.</param>
        /// <returns>This instance for fluent chaining.</returns>
        public ModelConstraints WithMaxMemory(long maxMemoryBytes)
        {
            if (maxMemoryBytes <= 0)
                throw new ArgumentException("Max memory must be positive", nameof(maxMemoryBytes));

            MaxMemory = maxMemoryBytes;
            return this;
        }

        /// <summary>
        /// Sets the minimum accuracy constraint.
        /// </summary>
        /// <param name="minAccuracy">Minimum accuracy (0.0 to 1.0).</param>
        /// <returns>This instance for fluent chaining.</returns>
        public ModelConstraints WithMinAccuracy(float minAccuracy)
        {
            if (minAccuracy < 0 || minAccuracy > 1)
                throw new ArgumentException("Min accuracy must be between 0 and 1", nameof(minAccuracy));

            MinAccuracy = minAccuracy;
            return this;
        }

        /// <summary>
        /// Sets the maximum file size constraint.
        /// </summary>
        /// <param name="maxFileSizeBytes">Maximum file size in bytes.</param>
        /// <returns>This instance for fluent chaining.</returns>
        public ModelConstraints WithMaxFileSize(long maxFileSizeBytes)
        {
            if (maxFileSizeBytes <= 0)
                throw new ArgumentException("Max file size must be positive", nameof(maxFileSizeBytes));

            MaxFileSize = maxFileSizeBytes;
            return this;
        }

        /// <summary>
        /// Sets the device type constraint.
        /// </summary>
        /// <param name="device">The device type.</param>
        /// <returns>This instance for fluent chaining.</returns>
        public ModelConstraints WithDevice(DeviceType device)
        {
            Device = device;
            return this;
        }

        /// <summary>
        /// Sets the batch size constraint.
        /// </summary>
        /// <param name="batchSize">The batch size.</param>
        /// <returns>This instance for fluent chaining.</returns>
        public ModelConstraints WithBatchSize(int batchSize)
        {
            if (batchSize <= 0)
                throw new ArgumentException("Batch size must be positive", nameof(batchSize));

            BatchSize = batchSize;
            return this;
        }

        /// <summary>
        /// Sets the deployment environment constraint.
        /// </summary>
        /// <param name="environment">The deployment environment.</param>
        /// <returns>This instance for fluent chaining.</returns>
        public ModelConstraints WithDeploymentEnvironment(DeploymentEnv environment)
        {
            DeploymentEnvironment = environment;
            return this;
        }
    }
}
