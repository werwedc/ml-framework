using System;

namespace MLFramework.ModelZoo.Discovery
{
    /// <summary>
    /// Provides memory usage estimation for model inference.
    /// </summary>
    public class MemoryEstimator
    {
        // Size of different data types in bytes
        private const float FLOAT32_SIZE = 4.0f;
        private const float FLOAT16_SIZE = 2.0f;
        private const float INT8_SIZE = 1.0f;

        // Device-specific memory overhead factors
        private const float CPU_MEMORY_OVERHEAD = 1.1f;    // 10% overhead
        private const float GPU_MEMORY_OVERHEAD = 1.2f;    // 20% overhead
        private const float TPU_MEMORY_OVERHEAD = 1.25f;   // 25% overhead
        private const float NPU_MEMORY_OVERHEAD = 1.15f;   // 15% overhead
        private const float EDGE_MEMORY_OVERHEAD = 1.05f;  // 5% overhead

        // Estimation of layers based on architecture
        private const int CNN_LAYERS_FACTOR = 50;
        private const int TRANSFORMER_LAYERS_FACTOR = 12;
        private const int MLP_LAYERS_FACTOR = 5;

        /// <summary>
        /// Estimates the memory usage for a model.
        /// </summary>
        /// <param name="model">The model metadata.</param>
        /// <param name="batchSize">The batch size.</param>
        /// <returns>Estimated memory usage in bytes.</returns>
        public float EstimateMemory(ModelMetadata model, int batchSize = 1)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            // Calculate model weights memory
            float weightsMemory = CalculateWeightsMemory(model);

            // Calculate activation memory
            float activationMemory = CalculateActivationMemory(model, batchSize);

            // Total memory = weights + activations
            float totalMemory = weightsMemory + activationMemory;

            // Apply device overhead
            float overhead = CPU_MEMORY_OVERHEAD; // Default to CPU

            return totalMemory * overhead;
        }

        /// <summary>
        /// Estimates the memory usage for a model on a specific device.
        /// </summary>
        /// <param name="model">The model metadata.</param>
        /// <param name="device">The target device type.</param>
        /// <param name="batchSize">The batch size.</param>
        /// <returns>Estimated memory usage in bytes.</returns>
        public float EstimateMemory(ModelMetadata model, DeviceType device, int batchSize = 1)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            // Calculate base memory
            float baseMemory = EstimateMemory(model, batchSize);

            // Get device-specific overhead
            float overhead = GetDeviceMemoryOverhead(device);

            return baseMemory * overhead;
        }

        /// <summary>
        /// Estimates memory usage with input shape consideration.
        /// </summary>
        /// <param name="model">The model metadata.</param>
        /// <param name="device">The target device type.</param>
        /// <param name="inputShape">The input shape.</param>
        /// <param name="batchSize">The batch size.</param>
        /// <returns>Estimated memory usage in bytes.</returns>
        public float EstimateMemoryWithShape(
            ModelMetadata model,
            DeviceType device,
            Shape inputShape,
            int batchSize = 1)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            // Calculate base memory
            float baseMemory = EstimateMemory(model, device, batchSize);

            // Adjust based on input shape
            if (inputShape != null)
            {
                long inputElements = inputShape.TotalElements();
                long expectedElements = model.InputShape.Length > 0
                    ? CalculateExpectedElements(model.InputShape)
                    : 1;

                // Adjust activation memory based on input size ratio
                if (expectedElements > 0)
                {
                    float sizeRatio = (float)inputElements / expectedElements;
                    float activationMemory = CalculateActivationMemory(model, batchSize);
                    float adjustedActivationMemory = activationMemory * sizeRatio;
                    float weightsMemory = CalculateWeightsMemory(model);

                    float overhead = GetDeviceMemoryOverhead(device);
                    baseMemory = (weightsMemory + adjustedActivationMemory) * overhead;
                }
            }

            return baseMemory;
        }

        /// <summary>
        /// Calculates the memory required for model weights.
        /// </summary>
        /// <param name="model">The model metadata.</param>
        /// <returns>Memory in bytes.</returns>
        private float CalculateWeightsMemory(ModelMetadata model)
        {
            // Assume float32 weights by default
            return model.NumParameters * FLOAT32_SIZE;
        }

        /// <summary>
        /// Calculates the memory required for activations during inference.
        /// </summary>
        /// <param name="model">The model metadata.</param>
        /// <param name="batchSize">The batch size.</param>
        /// <returns>Memory in bytes.</returns>
        private float CalculateActivationMemory(ModelMetadata model, int batchSize)
        {
            // Estimate number of layers based on architecture
            int estimatedLayers = EstimateLayerCount(model);

            // Calculate input/output dimensions
            long inputElements = model.InputShape.Length > 0
                ? CalculateExpectedElements(model.InputShape)
                : 1;

            long outputElements = model.OutputShape.Length > 0
                ? CalculateExpectedElements(model.OutputShape)
                : inputElements;

            // Activation memory estimation
            // Approximate as: batch_size * (input_elements + output_elements) * layers * sizeof(float)
            float activationMemory = batchSize * (inputElements + outputElements) * estimatedLayers * FLOAT32_SIZE;

            // Reduce by factor since not all layers store full activations
            activationMemory *= 0.5f;

            return activationMemory;
        }

        /// <summary>
        /// Estimates the number of layers in a model based on architecture.
        /// </summary>
        /// <param name="model">The model metadata.</param>
        /// <returns>Estimated layer count.</returns>
        private int EstimateLayerCount(ModelMetadata model)
        {
            if (string.IsNullOrWhiteSpace(model.Architecture))
                return 10; // Default estimate

            string archLower = model.Architecture.ToLowerInvariant();

            // CNN architectures
            if (archLower.Contains("resnet18") || archLower.Contains("resnet-18"))
                return 18;
            if (archLower.Contains("resnet34") || archLower.Contains("resnet-34"))
                return 34;
            if (archLower.Contains("resnet50") || archLower.Contains("resnet-50"))
                return 50;
            if (archLower.Contains("resnet101") || archLower.Contains("resnet-101"))
                return 101;
            if (archLower.Contains("resnet152") || archLower.Contains("resnet-152"))
                return 152;
            if (archLower.Contains("vgg16"))
                return 16;
            if (archLower.Contains("vgg19"))
                return 19;

            // Transformer architectures
            if (archLower.Contains("bert-base") || archLower.Contains("bert_base"))
                return 12;
            if (archLower.Contains("bert-large") || archLower.Contains("bert_large"))
                return 24;
            if (archLower.Contains("gpt-2") || archLower.Contains("gpt2"))
                return 12;

            // Use parameter-based estimation if specific model not recognized
            return (int)(model.NumParameters / 1000000) * 10; // ~10 layers per 1M parameters
        }

        /// <summary>
        /// Gets the device-specific memory overhead factor.
        /// </summary>
        /// <param name="device">The device type.</param>
        /// <returns>Overhead multiplier.</returns>
        private float GetDeviceMemoryOverhead(DeviceType device)
        {
            return device switch
            {
                DeviceType.CPU => CPU_MEMORY_OVERHEAD,
                DeviceType.GPU => GPU_MEMORY_OVERHEAD,
                DeviceType.TPU => TPU_MEMORY_OVERHEAD,
                DeviceType.NPU => NPU_MEMORY_OVERHEAD,
                DeviceType.Edge => EDGE_MEMORY_OVERHEAD,
                _ => CPU_MEMORY_OVERHEAD
            };
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
