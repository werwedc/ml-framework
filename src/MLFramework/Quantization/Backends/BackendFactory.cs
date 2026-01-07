using MLFramework.Quantization.Backends.CPUBackend;
using MLFramework.Quantization.Backends.x86Backend;
using MLFramework.Quantization.Backends.ARMBackend;
using MLFramework.Quantization.Backends.GPUBackend;

namespace MLFramework.Quantization.Backends
{
    /// <summary>
    /// Factory for creating quantization backend instances.
    /// </summary>
    public static class BackendFactory
    {
        private static readonly object _lock = new object();
        private static string? _preferredBackend;
        private static readonly Dictionary<string, IQuantizationBackend> _backendCache = new Dictionary<string, IQuantizationBackend>();

        /// <summary>
        /// Gets or sets the preferred backend name.
        /// </summary>
        public static string? PreferredBackend
        {
            get => _preferredBackend;
            set
            {
                lock (_lock)
                {
                    if (value != null && !GetAvailableBackends().Contains(value))
                    {
                        throw new ArgumentException($"Backend '{value}' is not available.", nameof(value));
                    }
                    _preferredBackend = value;
                }
            }
        }

        /// <summary>
        /// Creates the default backend based on system capabilities.
        /// </summary>
        /// <returns>The best available backend.</returns>
        public static IQuantizationBackend CreateDefault()
        {
            lock (_lock)
            {
                // If preferred backend is set and available, use it
                if (!string.IsNullOrEmpty(_preferredBackend))
                {
                    return Create(_preferredBackend!);
                }

                // Try backends in order of preference (best to worst)
                var availableBackends = GetAvailableBackends();

                // GPU is always preferred if available
                if (availableBackends.Contains("GPU"))
                {
                    return Create("GPU");
                }

                // Then x86 with oneDNN
                if (availableBackends.Contains("x86"))
                {
                    return Create("x86");
                }

                // Then ARM with NEON
                if (availableBackends.Contains("ARM"))
                {
                    return Create("ARM");
                }

                // Fallback to CPU (always available)
                return Create("CPU");
            }
        }

        /// <summary>
        /// Creates a specific backend by name.
        /// </summary>
        /// <param name="backendName">The name of the backend to create.</param>
        /// <returns>The backend instance.</returns>
        /// <exception cref="ArgumentException">Thrown when backend name is invalid.</exception>
        /// <exception cref="NotSupportedException">Thrown when backend is not available on the system.</exception>
        public static IQuantizationBackend Create(string backendName)
        {
            if (string.IsNullOrWhiteSpace(backendName))
            {
                throw new ArgumentException("Backend name cannot be null or whitespace.", nameof(backendName));
            }

            lock (_lock)
            {
                // Check cache first
                if (_backendCache.TryGetValue(backendName, out var cachedBackend))
                {
                    return cachedBackend;
                }

                IQuantizationBackend? backend = backendName.ToUpperInvariant() switch
                {
                    "CPU" => new CPUBackend.CPUBackend(),
                    "X86" => CreateX86Backend(),
                    "ARM" => CreateARMBackend(),
                    "GPU" => CreateGPUBackend(),
                    "INTEL ONEDNN" => CreateX86Backend(),
                    "ARM NEON" => CreateARMBackend(),
                    "CUDA" => CreateGPUBackend(),
                    _ => throw new ArgumentException($"Unknown backend: {backendName}", nameof(backendName))
                };

                if (backend == null || !backend.IsAvailable())
                {
                    throw new NotSupportedException($"Backend '{backendName}' is not available on this system.");
                }

                // Cache the backend
                _backendCache[backendName] = backend;

                return backend;
            }
        }

        /// <summary>
        /// Gets a list of all available backends on the current system.
        /// </summary>
        /// <returns>A list of available backend names.</returns>
        public static List<string> GetAvailableBackends()
        {
            var available = new List<string>();

            // CPU backend is always available
            available.Add("CPU");

            // Check x86 backend
            try
            {
                var x86Backend = CreateX86Backend();
                if (x86Backend != null && x86Backend.IsAvailable())
                {
                    available.Add("x86");
                }
            }
            catch
            {
                // x86 backend not available
            }

            // Check ARM backend
            try
            {
                var armBackend = CreateARMBackend();
                if (armBackend != null && armBackend.IsAvailable())
                {
                    available.Add("ARM");
                }
            }
            catch
            {
                // ARM backend not available
            }

            // Check GPU backend
            try
            {
                var gpuBackend = CreateGPUBackend();
                if (gpuBackend != null && gpuBackend.IsAvailable())
                {
                    available.Add("GPU");
                }
            }
            catch
            {
                // GPU backend not available
            }

            return available;
        }

        /// <summary>
        /// Clears the backend cache.
        /// </summary>
        public static void ClearCache()
        {
            lock (_lock)
            {
                _backendCache.Clear();
            }
        }

        /// <summary>
        /// Sets the preferred backend by name.
        /// </summary>
        /// <param name="backendName">The name of the preferred backend.</param>
        public static void SetPreferredBackend(string backendName)
        {
            PreferredBackend = backendName;
        }

        /// <summary>
        /// Creates an x86 backend if available.
        /// </summary>
        private static IQuantizationBackend? CreateX86Backend()
        {
            try
            {
                var backend = new x86Backend.x86Backend();
                return backend.IsAvailable() ? backend : null;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Creates an ARM backend if available.
        /// </summary>
        private static IQuantizationBackend? CreateARMBackend()
        {
            try
            {
                var backend = new ARMBackend.ARMBackend();
                return backend.IsAvailable() ? backend : null;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Creates a GPU backend if available.
        /// </summary>
        private static IQuantizationBackend? CreateGPUBackend()
        {
            try
            {
                var backend = new GPUBackend.GPUBackend();
                return backend.IsAvailable() ? backend : null;
            }
            catch
            {
                return null;
            }
        }
    }
}
