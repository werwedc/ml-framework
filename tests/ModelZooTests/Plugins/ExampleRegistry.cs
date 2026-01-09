using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;

namespace MLFramework.ModelZoo.Plugins
{
    /// <summary>
    /// Example HTTP-based registry plugin for demonstration purposes.
    /// This plugin demonstrates how to create a custom registry using the CustomModelRegistry base class.
    /// </summary>
    [RegistryPlugin("example", priority: 10)]
    public class ExampleRegistry : CustomModelRegistry
    {
        // Simple in-memory model database for demonstration
        private readonly Dictionary<string, ModelVersioning.ModelMetadata> _modelDatabase;
        private readonly Dictionary<string, byte[]> _modelData;

        public override string RegistryName => "example";
        public override int Priority => 10;

        public ExampleRegistry(PluginConfigurationBase configuration = null)
            : base(configuration ?? new PluginConfigurationBase())
        {
            _modelDatabase = new Dictionary<string, ModelVersioning.ModelMetadata>(StringComparer.OrdinalIgnoreCase);
            _modelData = new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase);

            // Initialize with some example models
            InitializeExampleModels();

            Console.WriteLine("[ExampleRegistry] Initialized with example models");
        }

        private void InitializeExampleModels()
        {
            // Add a simple neural network model
            var model1Data = new byte[] { 0x01, 0x02, 0x03, 0x04, 0x05 }; // Dummy data
            _modelData["simple-nn"] = model1Data;
            _modelDatabase["simple-nn"] = new ModelVersioning.ModelMetadata
            {
                ModelName = "simple-nn",
                Description = "A simple neural network model",
                Framework = "PyTorch",
                Architecture = "SimpleNN",
                InputShape = new[] { 784 },
                OutputShape = new[] { 10 },
                CustomMetadata = new Dictionary<string, string>
                {
                    { "Author", "Example Registry" },
                    { "Version", "1.0.0" },
                    { "Created", DateTime.UtcNow.ToString("O") }
                }
            };

            // Add a convolutional neural network model
            var model2Data = new byte[] { 0x10, 0x20, 0x30, 0x40, 0x50, 0x60 }; // Dummy data
            _modelData["convnet"] = model2Data;
            _modelDatabase["convnet"] = new ModelVersioning.ModelMetadata
            {
                ModelName = "convnet",
                Description = "A convolutional neural network for image classification",
                Framework = "TensorFlow",
                Architecture = "ConvNet",
                InputShape = new[] { 28, 28, 1 },
                OutputShape = new[] { 10 },
                CustomMetadata = new Dictionary<string, string>
                {
                    { "Author", "Example Registry" },
                    { "Version", "2.1.0" },
                    { "Created", DateTime.UtcNow.ToString("O") }
                }
            };

            // Add a transformer model
            var model3Data = new byte[] { 0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9 }; // Dummy data
            _modelData["transformer"] = model3Data;
            _modelDatabase["transformer"] = new ModelVersioning.ModelMetadata
            {
                ModelName = "transformer",
                Description = "A transformer model for NLP tasks",
                Framework = "PyTorch",
                Architecture = "Transformer",
                InputShape = new[] { 512 },
                OutputShape = new[] { 768 },
                CustomMetadata = new Dictionary<string, string>
                {
                    { "Author", "Example Registry" },
                    { "Version", "3.0.0" },
                    { "Created", DateTime.UtcNow.ToString("O") }
                }
            };
        }

        protected override async Task<ModelVersioning.ModelMetadata> FetchModelMetadataAsync(string modelName, string version = null)
        {
            await Task.Delay(50); // Simulate network delay

            if (_modelDatabase.TryGetValue(modelName, out var metadata))
            {
                LogInfo($"Fetched metadata for model: {modelName}");
                return metadata;
            }

            LogWarning($"Model not found: {modelName}");
            throw new KeyNotFoundException($"Model '{modelName}' not found in example registry");
        }

        protected override async Task<Stream> FetchModelAsync(ModelVersioning.ModelMetadata metadata, IProgress<double> progress = null)
        {
            if (metadata == null)
            {
                throw new ArgumentNullException(nameof(metadata));
            }

            var modelName = metadata.ModelName;

            if (!_modelData.ContainsKey(modelName))
            {
                throw new KeyNotFoundException($"Model data not found: {modelName}");
            }

            await Task.Delay(100); // Simulate download time

            // Report progress
            progress?.Report(0.3);
            await Task.Delay(50);
            progress?.Report(0.7);
            await Task.Delay(50);
            progress?.Report(1.0);

            var data = _modelData[modelName];
            LogInfo($"Downloaded model: {modelName} ({data.Length} bytes)");

            return new MemoryStream(data);
        }

        public override async Task<string[]> ListModelsAsync()
        {
            await Task.Delay(50); // Simulate network delay
            return _modelDatabase.Keys.ToArray();
        }

        public override bool CanHandle(string modelName)
        {
            // Handle models starting with "example:"
            if (modelName.StartsWith("example:", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            // Check if model exists in our database
            return _modelDatabase.ContainsKey(modelName);
        }

        /// <summary>
        /// Adds a custom model to the example registry (for testing).
        /// </summary>
        /// <param name="modelName">The model name.</param>
        /// <param name="metadata">The model metadata.</param>
        /// <param name="data">The model data.</param>
        public void AddCustomModel(string modelName, ModelVersioning.ModelMetadata metadata, byte[] data)
        {
            if (string.IsNullOrEmpty(modelName))
            {
                throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));
            }

            _modelDatabase[modelName] = metadata;
            _modelData[modelName] = data;
            LogInfo($"Added custom model: {modelName}");
        }

        /// <summary>
        /// Gets the number of models in the registry.
        /// </summary>
        public int ModelCount => _modelDatabase.Count;

        /// <summary>
        /// Checks if a model exists in the registry.
        /// </summary>
        /// <param name="modelName">The model name.</param>
        /// <returns>True if the model exists, false otherwise.</returns>
        public bool HasModel(string modelName)
        {
            return _modelDatabase.ContainsKey(modelName);
        }
    }
}
