using System.Runtime.InteropServices;

namespace MLFramework.Quantization.Backends.GPUBackend
{
    /// <summary>
    /// CUDA feature detection for GPU quantization support.
    /// </summary>
    internal static class CUDAFeatureDetection
    {
        private static bool? _isAvailable;
        private static int? _maxThreads;

        /// <summary>
        /// Checks if CUDA is available.
        /// </summary>
        /// <returns>True if CUDA is available, false otherwise.</returns>
        public static bool IsAvailable()
        {
            if (_isAvailable.HasValue)
            {
                return _isAvailable.Value;
            }

            // Check if we're on Windows or Linux
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ||
                RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                // Try to detect CUDA by checking for nvcuda.dll or libcuda.so
                try
                {
                    if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                    {
                        // Check for CUDA on Windows
                        _isAvailable = TryLoadCudaWindows();
                    }
                    else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                    {
                        // Check for CUDA on Linux
                        _isAvailable = TryLoadCudaLinux();
                    }
                    else
                    {
                        _isAvailable = false;
                    }
                }
                catch
                {
                    _isAvailable = false;
                }
            }
            else
            {
                _isAvailable = false;
            }

            return _isAvailable.Value;
        }

        /// <summary>
        /// Gets the maximum number of threads supported by CUDA.
        /// </summary>
        /// <returns>The maximum number of threads.</returns>
        public static int GetMaxThreads()
        {
            if (_maxThreads.HasValue)
            {
                return _maxThreads.Value;
            }

            if (!IsAvailable())
            {
                _maxThreads = Environment.ProcessorCount;
                return _maxThreads.Value;
            }

            // TODO: Query CUDA device for max threads
            // For now, return a reasonable default
            _maxThreads = 1024;
            return _maxThreads.Value;
        }

        /// <summary>
        /// Gets the detected CUDA device information.
        /// </summary>
        /// <returns>A string describing the detected CUDA device.</returns>
        public static string GetDeviceInfo()
        {
            if (!IsAvailable())
            {
                return "No CUDA device detected";
            }

            // TODO: Query CUDA device for actual information
            return "CUDA device detected (information not yet implemented)";
        }

        /// <summary>
        /// Tries to load CUDA on Windows.
        /// </summary>
        private static bool TryLoadCudaWindows()
        {
            try
            {
                // Try to load nvcuda.dll
                var cudaLib = LoadLibrary("nvcuda.dll");
                if (cudaLib != IntPtr.Zero)
                {
                    FreeLibrary(cudaLib);
                    return true;
                }
            }
            catch
            {
                // Failed to load CUDA
            }

            return false;
        }

        /// <summary>
        /// Tries to load CUDA on Linux.
        /// </summary>
        private static bool TryLoadCudaLinux()
        {
            try
            {
                // Check if libcuda.so exists
                var libPath = "/usr/lib/x86_64-linux-gnu/libcuda.so";
                if (File.Exists(libPath))
                {
                    return true;
                }

                libPath = "/usr/local/cuda/lib64/libcuda.so";
                if (File.Exists(libPath))
                {
                    return true;
                }
            }
            catch
            {
                // Failed to check for CUDA
            }

            return false;
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr LoadLibrary(string lpFileName);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool FreeLibrary(IntPtr hModule);
    }
}
