using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.IR.Backend
{
    /// <summary>
    /// Registry for managing compilation backends
    /// </summary>
    public class BackendRegistry
    {
        private static BackendRegistry _instance;
        private Dictionary<string, IBackend> _backends;

        private BackendRegistry()
        {
            _backends = new Dictionary<string, IBackend>(StringComparer.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Singleton instance of the backend registry
        /// </summary>
        public static BackendRegistry Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = new BackendRegistry();
                }
                return _instance;
            }
        }

        /// <summary>
        /// Register a backend
        /// </summary>
        public void RegisterBackend(IBackend backend)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            _backends[backend.Name] = backend;
        }

        /// <summary>
        /// Get a backend by name
        /// </summary>
        public IBackend GetBackend(string name)
        {
            if (name == null)
                throw new ArgumentNullException(nameof(name));

            return _backends.TryGetValue(name, out var backend) ? backend : null;
        }

        /// <summary>
        /// Get all registered backends
        /// </summary>
        public IEnumerable<IBackend> GetAllBackends() => _backends.Values;

        /// <summary>
        /// Check if a backend is registered
        /// </summary>
        public bool HasBackend(string name)
        {
            if (name == null)
                throw new ArgumentNullException(nameof(name));

            return _backends.ContainsKey(name);
        }

        /// <summary>
        /// Unregister a backend
        /// </summary>
        public bool UnregisterBackend(string name)
        {
            if (name == null)
                throw new ArgumentNullException(nameof(name));

            return _backends.Remove(name);
        }

        /// <summary>
        /// Clear all registered backends
        /// </summary>
        public void Clear()
        {
            _backends.Clear();
        }
    }
}
