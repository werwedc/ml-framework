using System;

namespace MLFramework.ModelZoo.Discovery
{
    /// <summary>
    /// Provides latency estimation for model inference based on model characteristics and device capabilities.
    /// </summary>
    public class LatencyEstimator
    {
        // Base FLOPS per second for different device types (in billions)
        private const float CPU_FLOPS = 0.1f;      // 100 MFLOPS
        private const float GPU_FLOPS = 10.0f;     // 10 GFLOPS
        private const float TPU_FLOPS = 100.0f;    // 100 GFLOPS
        private const float NPU_FLOPS = 50.0f;     // 50 GFLOPS
        private const float EDGE_FLOPS = 0.05f;   // 50 MFLOPS

        // Architecture-specific multipliers
        private const float CNN_MULTIPLIER = 1.0f;
        private const float TRANSFORMER_MULTIPLIER = 1.5f;
        private const float MLP_MULTIPLIER = 0.8f;

        /// <summary>
        /// Estimates the inference latency for a model.
        /// </summary>
        /// <param name="model">The model metadata.</param>
        /// <param name="device">The target device type.</param>
        /// <param name="batchSize">The batch size.</param>
        /// <returns>Estimated latency in milliseconds.</returns>
        public float EstimateLatency(ModelMetadata model, DeviceType device, int batchSize = 1)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            // Get device FLOPS
            float deviceFlops = GetDeviceFlops(device);

            // Get architecture multiplier
            float archMultiplier = GetArchitectureMultiplier(model.Architecture);

            // Calculate base operations (using parameter count as proxy)
            // This is a simplified model - real latency depends on actual FLOPS, not just parameters
            float baseOperations = model.NumParameters * 2; // Each parameter typically involves ~2 operations

            // Adjust for batch size
            float batchedOperations = baseOperations * batchSize;

            // Adjust for architecture complexity
            float adjustedOperations = batchedOperations * archMultiplier;

            // Calculate latency in seconds
            float latencySeconds = adjustedOperations / (deviceFlops * 1e9f);

            // Convert to milliseconds
            float latencyMs = latencySeconds * 1000;

            // Apply some adjustments for more realistic estimates
            latencyMs = ApplyRealisticAdjustments(latencyMs, model, device);

            return latencyMs;
        }

        /// <summary>
        /// Estimates latency with input shape consideration.
        /// </summary>
        /// <param name="model">The model metadata.</param>
        /// <param name="device">The target device type.</param>
        /// <param name="inputShape">The input shape.</param>
        /// <param name="batchSize">The batch size.</param>
        /// <returns>Estimated latency in milliseconds.</returns>
        public float EstimateLatencyWithShape(
            ModelMetadata model,
            DeviceType device,
            Shape inputShape,
            int batchSize = 1)
        {
            float baseLatency = EstimateLatency(model, device, batchSize);

            // Adjust based on input shape size
            if (inputShape != null)
            {
                long inputElements = inputShape.TotalElements();
                long expectedElements = model.InputShape.Length > 0
                    ? CalculateExpectedElements(model.InputShape)
                    : 1;

                // Scale latency based on input size ratio
                if (expectedElements > 0)
                {
                    float sizeRatio = (float)inputElements / expectedElements;
                    baseLatency *= Math.Sqrt(sizeRatio); // Use sqrt to account for non-linear scaling
                }
            }

            return baseLatency;
        }

        /// <summary>
        /// Gets the FLOPS capacity for a device type.
        /// </summary>
        /// <param name="device">The device type.</param>
        /// <returns>FLOPS in billions per second.</returns>
        private float GetDeviceFlops(DeviceType device)
        {
            return device switch
            {
                DeviceType.CPU => CPU_FLOPS,
                DeviceType.GPU => GPU_FLOPS,
                DeviceType.TPU => TPU_FLOPS,
                DeviceType.NPU => NPU_FLOPS,
                DeviceType.Edge => EDGE_FLOPS,
                _ => CPU_FLOPS
            };
        }

        /// <summary>
        /// Gets the architecture complexity multiplier.
        /// </summary>
        /// <param name="architecture">The architecture name.</param>
        /// <returns>Multiplier for architecture type.</returns>
        private float GetArchitectureMultiplier(string architecture)
        {
            if (string.IsNullOrWhiteSpace(architecture))
                return 1.0f;

            string archLower = architecture.ToLowerInvariant();

            // CNN architectures
            if (archLower.Contains("resnet") ||
                archLower.Contains("vgg") ||
                archLower.Contains("efficientnet") ||
                archLower.Contains("mobilenet") ||
                archLower.Contains("conv"))
            {
                return CNN_MULTIPLIER;
            }

            // Transformer architectures
            if (archLower.Contains("bert") ||
                archLower.Contains("gpt") ||
                archLower.Contains("transformer") ||
                archLower.Contains("attention"))
            {
                return TRANSFORMER_MULTIPLIER;
            }

            // MLP architectures
            if (archLower.Contains("mlp") ||
                archLower.Contains("dense") ||
                archLower.Contains("linear"))
            {
                return MLP_MULTIPLIER;
            }

            return 1.0f;
        }

        /// <summary>
        /// Applies realistic adjustments to the base latency estimate.
        /// </summary>
        /// <param name="baseLatency">The base latency estimate.</param>
        /// <param name="model">The model metadata.</param>
        /// <param name="device">The device type.</param>
        /// <returns>Adjusted latency in milliseconds.</returns>
        private float ApplyRealisticAdjustments(float baseLatency, ModelMetadata model, DeviceType device)
        {
            // Add a small overhead for model loading and preprocessing
            float overhead = 1.0f; // 1ms overhead

            // Adjust for device-specific considerations
            switch (device)
            {
                case DeviceType.CPU:
                    // CPUs have higher memory access overhead
                    overhead += 2.0f;
                    break;
                case DeviceType.GPU:
                case DeviceType.TPU:
                case DeviceType.NPU:
                    // Accelerators have better parallelization
                    overhead += 0.5f;
                    break;
                case DeviceType.Edge:
                    // Edge devices have higher overhead
                    overhead += 3.0f;
                    break;
            }

            // Ensure minimum latency
            float minLatency = 0.1f; // 0.1ms minimum
            return Math.Max(baseLatency + overhead, minLatency);
        }

        /// <summary>
        /// Calculates expected number of input elements.
        /// </summary>
        /// <param name="inputShape">The input shape array.</param>
        /// <returns>Total number of elements.</returns>
        private long CalculateExpectedElements(int[] inputShape)
        {
            if (inputShape == null || inputShape.Length == 0)
                return 1;

            long total = 1;
            foreach (int dim in inputShape)
            {
                if (dim > 0)
                    total *= dim;
            }
            return total;
        }
    }
}
