using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace MLFramework.ModelZoo.Plugins.BuiltIn
{
    /// <summary>
    /// Local file system registry plugin.
    /// Loads models from local file system.
    /// </summary>
    [RegistryPlugin("local", priority: 100)]
    public class LocalFileRegistry : CustomModelRegistry
    {
        private readonly string _baseDirectory;

        public override string RegistryName => "local";
        public override int Priority => 100;

        public LocalFileRegistry(string baseDirectory, PluginConfigurationBase configuration = null)
            : base(configuration ?? new PluginConfigurationBase())
        {
            if (string.IsNullOrEmpty(baseDirectory))
            {
                throw new ArgumentException("Base directory cannot be null or empty", nameof(baseDirectory));
            }

            if (!Directory.Exists(baseDirectory))
            {
                Directory.CreateDirectory(baseDirectory);
            }

            _baseDirectory = Path.GetFullPath(baseDirectory);
            LogInfo($"Initialized with base directory: {_baseDirectory}");
        }

        protected override async Task<ModelVersioning.ModelMetadata> FetchModelMetadataAsync(string modelName, string version = null)
        {
            var modelPath = GetModelPath(modelName, version);

            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"Model not found: {modelPath}");
            }

            // Create metadata from file info
            var fileInfo = new FileInfo(modelPath);
            var metadata = new ModelVersioning.ModelMetadata
            {
                ModelName = modelName,
                Description = $"Model from local file system: {modelPath}",
                Framework = Path.GetExtension(modelPath).TrimStart('.').ToUpperInvariant(),
                Architecture = "Local",
                CustomMetadata = new System.Collections.Generic.Dictionary<string, string>
                {
                    { "FilePath", modelPath },
                    { "FileSize", fileInfo.Length.ToString() },
                    { "LastModified", fileInfo.LastWriteTimeUtc.ToString("O") }
                }
            };

            return await Task.FromResult(metadata);
        }

        protected override async Task<Stream> FetchModelAsync(ModelVersioning.ModelMetadata metadata, IProgress<double> progress = null)
        {
            var filePath = metadata.CustomMetadata?.GetValueOrDefault("FilePath");

            if (string.IsNullOrEmpty(filePath))
            {
                throw new InvalidOperationException("Model file path not found in metadata");
            }

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"Model file not found: {filePath}");
            }

            LogInfo($"Reading model from: {filePath}");

            // Read file with progress reporting
            var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            var buffer = new byte[81920]; // 80KB buffer
            var memoryStream = new MemoryStream();

            long totalBytes = fileStream.Length;
            long bytesRead = 0;

            int read;
            while ((read = await fileStream.ReadAsync(buffer, 0, buffer.Length)) > 0)
            {
                await memoryStream.WriteAsync(buffer, 0, read);
                bytesRead += read;

                progress?.Report((double)bytesRead / totalBytes);
            }

            fileStream.Dispose();
            memoryStream.Position = 0;

            LogInfo($"Successfully read {totalBytes} bytes");
            return memoryStream;
        }

        public override async Task<string[]> ListModelsAsync()
        {
            if (!Directory.Exists(_baseDirectory))
            {
                return await Task.FromResult(Array.Empty<string>());
            }

            var models = Directory.GetFiles(_baseDirectory, "*.*", SearchOption.AllDirectories)
                .Select(f => Path.GetFileNameWithoutExtension(f))
                .Distinct()
                .ToArray();

            return await Task.FromResult(models);
        }

        public override bool CanHandle(string modelName)
        {
            // Handle models starting with "local:" or in the local directory
            if (modelName.StartsWith("local:", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            var modelPath = GetModelPath(modelName);
            return File.Exists(modelPath);
        }

        private string GetModelPath(string modelName, string version = null)
        {
            // Remove "local:" prefix if present
            if (modelName.StartsWith("local:", StringComparison.OrdinalIgnoreCase))
            {
                modelName = modelName.Substring(6);
            }

            // Try common model file extensions
            var extensions = new[] { ".bin", ".pt", ".pth", ".pkl", ".onnx", ".pb", ".h5", "" };

            foreach (var ext in extensions)
            {
                var path = Path.Combine(_baseDirectory, modelName + ext);
                if (File.Exists(path))
                {
                    return path;
                }
            }

            // If not found, return the path with default extension
            return Path.Combine(_baseDirectory, modelName + ".bin");
        }

        public new void Dispose()
        {
            base.Dispose();
        }
    }
}
