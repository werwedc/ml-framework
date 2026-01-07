using System.Collections.Concurrent;

namespace RitterFramework.Core.LoRA
{
    /// <summary>
    /// Manager for loading, saving, switching, and managing multiple LoRA adapters on a model.
    /// Supports single or multiple active adapters simultaneously.
    /// </summary>
    public class AdapterManager
    {
        private readonly IModule _baseModel;
        private readonly ConcurrentDictionary<string, LoraAdapter> _loadedAdapters;
        private readonly ConcurrentDictionary<string, bool> _activeAdapters;
        private readonly LoraConfig _defaultConfig;
        private readonly object _lock = new object();

        /// <summary>
        /// Constructor.
        /// </summary>
        public AdapterManager(IModule baseModel, LoraConfig? config = null)
        {
            _baseModel = baseModel ?? throw new ArgumentNullException(nameof(baseModel));
            _defaultConfig = config ?? new LoraConfig();
            _loadedAdapters = new ConcurrentDictionary<string, LoraAdapter>();
            _activeAdapters = new ConcurrentDictionary<string, bool>();
        }

        /// <summary>
        /// Load adapter from file.
        /// </summary>
        public void LoadAdapter(string name, string path)
        {
            if (string.IsNullOrEmpty(name)) throw new ArgumentException("Name cannot be null or empty", nameof(name));
            if (string.IsNullOrEmpty(path)) throw new ArgumentException("Path cannot be null or empty", nameof(path));

            var adapter = AdapterSerializer.Load(path);

            // Validate adapter configuration matches expected configuration
            if (_defaultConfig.Rank != 0 && adapter.Config.Rank != _defaultConfig.Rank)
            {
                throw new InvalidOperationException(
                    $"Adapter rank mismatch. Expected: {_defaultConfig.Rank}, Got: {adapter.Config.Rank}");
            }

            _loadedAdapters[name] = adapter;
        }

        /// <summary>
        /// Load adapter from LoraAdapter object.
        /// </summary>
        public void LoadAdapter(LoraAdapter adapter)
        {
            if (adapter == null) throw new ArgumentNullException(nameof(adapter));
            if (string.IsNullOrEmpty(adapter.Name))
            {
                throw new ArgumentException("Adapter name cannot be null or empty", nameof(adapter));
            }

            _loadedAdapters[adapter.Name] = adapter;
        }

        /// <summary>
        /// Save adapter to disk.
        /// </summary>
        public void SaveAdapter(string name, string path)
        {
            if (string.IsNullOrEmpty(name)) throw new ArgumentException("Name cannot be null or empty", nameof(name));
            if (string.IsNullOrEmpty(path)) throw new ArgumentException("Path cannot be null or empty", nameof(path));

            if (!_loadedAdapters.TryGetValue(name, out var adapter))
            {
                throw new ArgumentException($"Adapter '{name}' not loaded", nameof(name));
            }

            // Extract current weights from model
            var currentWeights = ExtractAdapterWeights();
            adapter.Weights = currentWeights;

            AdapterSerializer.Save(adapter, path);
        }

        /// <summary>
        /// Save adapter to JSON format.
        /// </summary>
        public void SaveAdapterJson(string name, string path)
        {
            if (string.IsNullOrEmpty(name)) throw new ArgumentException("Name cannot be null or empty", nameof(name));
            if (string.IsNullOrEmpty(path)) throw new ArgumentException("Path cannot be null or empty", nameof(path));

            if (!_loadedAdapters.TryGetValue(name, out var adapter))
            {
                throw new ArgumentException($"Adapter '{name}' not loaded", nameof(name));
            }

            // Extract current weights from model
            var currentWeights = ExtractAdapterWeights();
            adapter.Weights = currentWeights;

            AdapterSerializer.SaveJson(adapter, path);
        }

        /// <summary>
        /// Set active adapter(s) - replaces current active adapters.
        /// </summary>
        public void SetActiveAdapter(params string[] names)
        {
            lock (_lock)
            {
                // Validate all adapters are loaded
                foreach (var name in names)
                {
                    if (string.IsNullOrEmpty(name))
                    {
                        throw new ArgumentException("Adapter name cannot be null or empty");
                    }

                    if (!_loadedAdapters.ContainsKey(name))
                    {
                        throw new ArgumentException($"Adapter '{name}' not loaded");
                    }
                }

                // Clear current active adapters
                _activeAdapters.Clear();

                // Set new active adapters
                foreach (var name in names)
                {
                    _activeAdapters[name] = true;
                }

                // Apply active adapters to model
                ApplyActiveAdapters();
            }
        }

        /// <summary>
        /// Add adapter to active set (multi-adapter support).
        /// </summary>
        public void ActivateAdapter(string name)
        {
            if (string.IsNullOrEmpty(name)) throw new ArgumentException("Name cannot be null or empty", nameof(name));

            lock (_lock)
            {
                if (!_loadedAdapters.ContainsKey(name))
                {
                    throw new ArgumentException($"Adapter '{name}' not loaded");
                }

                _activeAdapters[name] = true;
                ApplyActiveAdapters();
            }
        }

        /// <summary>
        /// Remove adapter from active set.
        /// </summary>
        public void DeactivateAdapter(string name)
        {
            if (string.IsNullOrEmpty(name)) throw new ArgumentException("Name cannot be null or empty", nameof(name));

            lock (_lock)
            {
                _activeAdapters.TryRemove(name, out _);
                ApplyActiveAdapters();
            }
        }

        /// <summary>
        /// List all loaded adapter names.
        /// </summary>
        public IReadOnlyList<string> ListAdapters()
        {
            return _loadedAdapters.Keys.ToList().AsReadOnly();
        }

        /// <summary>
        /// List active adapter names.
        /// </summary>
        public IReadOnlyList<string> ListActiveAdapters()
        {
            return _activeAdapters.Keys.ToList().AsReadOnly();
        }

        /// <summary>
        /// Unload adapter from memory.
        /// </summary>
        public void UnloadAdapter(string name)
        {
            if (string.IsNullOrEmpty(name)) throw new ArgumentException("Name cannot be null or empty", nameof(name));

            lock (_lock)
            {
                _activeAdapters.TryRemove(name, out _);
                _loadedAdapters.TryRemove(name, out _);
                ApplyActiveAdapters();
            }
        }

        /// <summary>
        /// Get adapter by name.
        /// </summary>
        public LoraAdapter GetAdapter(string name)
        {
            if (string.IsNullOrEmpty(name)) throw new ArgumentException("Name cannot be null or empty", nameof(name));

            if (!_loadedAdapters.TryGetValue(name, out var adapter))
            {
                throw new ArgumentException($"Adapter '{name}' not loaded", nameof(name));
            }

            return adapter;
        }

        /// <summary>
        /// Get the base model.
        /// </summary>
        public IModule BaseModel => _baseModel;

        private void ApplyActiveAdapters()
        {
            // Apply active adapter weights to model's LoRA layers
            foreach (var loraLayer in FindLoRALayers(_baseModel))
            {
                var moduleName = GetModuleName(loraLayer);

                // Reset to zeros first
                loraLayer.ResetLoRA();

                // Accumulate weights from all active adapters
                foreach (var adapterName in _activeAdapters.Keys)
                {
                    if (_loadedAdapters.TryGetValue(adapterName, out var adapter))
                    {
                        if (adapter.TryGetModuleWeights(moduleName, out var weights))
                        {
                            loraLayer.AddLoRAWeights(weights.LoraA, weights.LoraB);
                        }
                    }
                }
            }
        }

        private IEnumerable<LoraLinear> FindLoRALayers(IModule model)
        {
            // Recursively find all LoraLinear layers in the model
            // This is a simplified implementation that checks if the module itself is a LoraLinear
            if (model is LoraLinear loraLinear)
            {
                yield return loraLinear;
            }

            // Note: In a full implementation, this would also recursively search
            // nested modules/containers. This would require a module traversal API.
        }

        private string GetModuleName(LoraLinear layer)
        {
            return layer.Name ?? "unknown";
        }

        private Dictionary<string, LoraModuleWeights> ExtractAdapterWeights()
        {
            var weights = new Dictionary<string, LoraModuleWeights>();

            foreach (var loraLayer in FindLoRALayers(_baseModel))
            {
                weights[GetModuleName(loraLayer)] = loraLayer.GetLoRAWeights();
            }

            return weights;
        }
    }
}
