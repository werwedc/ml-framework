using MLFramework.Distributed.Gloo;
using System;
using System.Runtime.InteropServices;

namespace MLFramework.Distributed.Gloo
{
    /// <summary>
    /// Gloo backend for distributed communication.
    /// Gloo provides CPU and multi-GPU communication, and works on both Linux and Windows.
    /// </summary>
    public class GlooBackend : ICommunicationBackend, IDisposable
    {
        private IntPtr _context;  // gloo::rendezvous::Context
        private int _rank;
        private int _worldSize;
        private bool _initialized;
        private bool _disposed;

        /// <summary>
        /// Gets the name of the backend.
        /// </summary>
        public string Name => "Gloo";

        /// <summary>
        /// Gets whether Gloo is available on this system.
        /// </summary>
        public bool IsAvailable => CheckAvailability();

        /// <summary>
        /// Gets the number of devices available for communication.
        /// For Gloo, this is the world size.
        /// </summary>
        public int DeviceCount => _worldSize;

        /// <summary>
        /// Gets whether the backend supports asynchronous operations.
        /// Gloo's async support is limited.
        /// </summary>
        public bool SupportsAsync => false;

        /// <summary>
        /// Gets whether the backend supports GPU direct communication.
        /// Gloo does not support GPUDirect.
        /// </summary>
        public bool SupportsGPUDirect => false;

        /// <summary>
        /// Gets the buffer size limit for communication operations.
        /// Default is 1GB.
        /// </summary>
        public long GetBufferSizeLimit() => 1024 * 1024 * 1024;

        /// <summary>
        /// Gets the rank of this process (for internal use by process group).
        /// </summary>
        internal int Rank => _rank;

        /// <summary>
        /// Gets the world size (for internal use by process group).
        /// </summary>
        internal int WorldSize => _worldSize;

        /// <summary>
        /// Gets the Gloo context (for internal use).
        /// </summary>
        internal IntPtr Context => _context;

        public GlooBackend()
        {
            _initialized = false;
            _context = IntPtr.Zero;
            _disposed = false;
        }

        /// <summary>
        /// Initialize Gloo context.
        /// Must be called before any communication operations.
        /// </summary>
        public void Initialize()
        {
            if (_initialized)
            {
                throw new InvalidOperationException("Gloo backend is already initialized");
            }

            try
            {
                // Read environment variables
                _rank = GetEnvVar("RANK", 0);
                _worldSize = GetEnvVar("WORLD_SIZE", 1);
                var iface = GetEnvVar("GLOO_IFACE", "eth0");
                var transport = GetEnvVar("GLOO_DEVICE_TRANSPORT", "tcp");

                // Create Gloo context
                _context = GlooNative.gloo_create_context(_rank, _worldSize, iface, transport);

                if (_context == IntPtr.Zero)
                {
                    throw new CommunicationException("Failed to create Gloo context", _rank, Name);
                }

                _initialized = true;
            }
            catch (DllNotFoundException)
            {
                throw new CommunicationException("Gloo library not found", _rank, Name);
            }
            catch (Exception ex)
            {
                throw new CommunicationException($"Failed to initialize Gloo backend: {ex.Message}", _rank, Name, ex);
            }
        }

        /// <summary>
        /// Finalize Gloo context and free resources.
        /// </summary>
        private void Cleanup()
        {
            if (_initialized && _context != IntPtr.Zero)
            {
                try
                {
                    GlooNative.gloo_destroy_context(_context);
                }
                catch (Exception)
                {
                    // Ignore errors during cleanup
                }

                _context = IntPtr.Zero;
                _initialized = false;
            }
        }

        /// <summary>
        /// Check if Gloo is available on this system.
        /// </summary>
        public static bool CheckAvailability()
        {
            try
            {
                // Try to load Gloo library
                var handle = LoadLibrary("gloo");
                if (handle == IntPtr.Zero)
                {
                    return false;
                }

                FreeLibrary(handle);
                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Get integer environment variable or default value.
        /// </summary>
        private static int GetEnvVar(string name, int defaultValue)
        {
            var value = Environment.GetEnvironmentVariable(name);
            if (int.TryParse(value, out int result))
            {
                return result;
            }
            return defaultValue;
        }

        /// <summary>
        /// Get string environment variable or default value.
        /// </summary>
        private static string GetEnvVar(string name, string defaultValue)
        {
            return Environment.GetEnvironmentVariable(name) ?? defaultValue;
        }

        /// <summary>
        /// P/Invoke for LoadLibrary.
        /// </summary>
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr LoadLibrary([MarshalAs(UnmanagedType.LPStr)] string lpFileName);

        /// <summary>
        /// P/Invoke for FreeLibrary.
        /// </summary>
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool FreeLibrary(IntPtr hModule);

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    Cleanup();
                }
                _disposed = true;
            }
        }

        ~GlooBackend()
        {
            Dispose(false);
        }
    }
}
