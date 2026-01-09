using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Helper class to build ModelRegistry instances from various sources.
    /// </summary>
    public class RegistryBuilder
    {
        private readonly List<ModelMetadata> _metadataList;

        /// <summary>
        /// Initializes a new instance of the RegistryBuilder class.
        /// </summary>
        public RegistryBuilder()
        {
            _metadataList = new List<ModelMetadata>();
        }

        /// <summary>
        /// Loads all metadata JSON files from a directory.
        /// </summary>
        /// <param name="path">The directory path.</param>
        /// <returns>The builder instance for method chaining.</returns>
        /// <exception cref="ArgumentNullException">Thrown when path is null or empty.</exception>
        /// <exception cref="DirectoryNotFoundException">Thrown when the directory does not exist.</exception>
        public RegistryBuilder AddFromDirectory(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentNullException(nameof(path));

            if (!Directory.Exists(path))
                throw new DirectoryNotFoundException($"Directory not found: {path}");

            var jsonFiles = Directory.GetFiles(path, "*.json", SearchOption.TopDirectoryOnly);

            foreach (var file in jsonFiles)
            {
                try
                {
                    var metadata = ModelMetadata.LoadFromJsonFile(file);
                    _metadataList.Add(metadata);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Failed to load metadata from file: {file}", ex);
                }
            }

            return this;
        }

        /// <summary>
        /// Loads metadata from a single JSON file.
        /// </summary>
        /// <param name="path">The file path.</param>
        /// <returns>The builder instance for method chaining.</returns>
        /// <exception cref="ArgumentNullException">Thrown when path is null or empty.</exception>
        /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
        public RegistryBuilder AddFromJsonFile(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentNullException(nameof(path));

            if (!File.Exists(path))
                throw new FileNotFoundException($"File not found: {path}");

            var metadata = ModelMetadata.LoadFromJsonFile(path);
            _metadataList.Add(metadata);

            return this;
        }

        /// <summary>
        /// Loads metadata from an embedded resource in the assembly.
        /// </summary>
        /// <param name="resourceName">The full resource name.</param>
        /// <param name="assembly">The assembly containing the resource (null for calling assembly).</param>
        /// <returns>The builder instance for method chaining.</returns>
        /// <exception cref="ArgumentNullException">Thrown when resourceName is null or empty.</exception>
        public RegistryBuilder AddFromEmbeddedResource(string resourceName, Assembly? assembly = null)
        {
            if (string.IsNullOrWhiteSpace(resourceName))
                throw new ArgumentNullException(nameof(resourceName));

            assembly ??= Assembly.GetCallingAssembly();

            using (var stream = assembly.GetManifestResourceStream(resourceName))
            {
                if (stream == null)
                    throw new FileNotFoundException($"Embedded resource not found: {resourceName}");

                using (var reader = new StreamReader(stream))
                {
                    string json = reader.ReadToEnd();
                    var metadata = ModelMetadata.FromJson(json);
                    _metadataList.Add(metadata);
                }
            }

            return this;
        }

        /// <summary>
        /// Adds metadata directly to the builder.
        /// </summary>
        /// <param name="metadata">The metadata to add.</param>
        /// <returns>The builder instance for method chaining.</returns>
        /// <exception cref="ArgumentNullException">Thrown when metadata is null.</exception>
        public RegistryBuilder Add(ModelMetadata metadata)
        {
            if (metadata == null)
                throw new ArgumentNullException(nameof(metadata));

            _metadataList.Add(metadata);
            return this;
        }

        /// <summary>
        /// Adds multiple metadata objects to the builder.
        /// </summary>
        /// <param name="metadataList">The list of metadata to add.</param>
        /// <returns>The builder instance for method chaining.</returns>
        /// <exception cref="ArgumentNullException">Thrown when metadataList is null.</exception>
        public RegistryBuilder AddRange(IEnumerable<ModelMetadata> metadataList)
        {
            if (metadataList == null)
                throw new ArgumentNullException(nameof(metadataList));

            _metadataList.AddRange(metadataList);
            return this;
        }

        /// <summary>
        /// Constructs the final ModelRegistry with all added metadata.
        /// </summary>
        /// <returns>A new ModelRegistry instance populated with all metadata.</returns>
        public ModelRegistry Build()
        {
            var registry = new ModelRegistry();
            foreach (var metadata in _metadataList)
            {
                try
                {
                    registry.Register(metadata);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException(
                        $"Failed to register model '{metadata.Name}' version '{metadata.Version}' in registry", ex);
                }
            }
            return registry;
        }

        /// <summary>
        /// Gets the number of metadata objects currently in the builder.
        /// </summary>
        public int Count => _metadataList.Count;

        /// <summary>
        /// Clears all metadata from the builder.
        /// </summary>
        public void Clear()
        {
            _metadataList.Clear();
        }
    }
}
