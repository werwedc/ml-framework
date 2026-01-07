using System.Runtime.Intrinsics.X86;

namespace MLFramework.Quantization.Backends.x86Backend
{
    /// <summary>
    /// x86 CPU feature detection for oneDNN, AVX-512, and VNNI support.
    /// </summary>
    internal static class x86FeatureDetection
    {
        private static bool? _isAvailable;

        /// <summary>
        /// Checks if x86 features are available.
        /// </summary>
        /// <returns>True if x86 features are available, false otherwise.</returns>
        public static bool IsAvailable()
        {
            if (_isAvailable.HasValue)
            {
                return _isAvailable.Value;
            }

            // Check if we're on an x86/x64 platform
            if (Environment.Is64BitProcess && (Environment.OSVersion.Platform == PlatformID.Win32NT ||
                System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux)))
            {
                // Check for AVX2 (minimum requirement for oneDNN)
                _isAvailable = Avx2.IsSupported;
            }
            else
            {
                _isAvailable = false;
            }

            return _isAvailable.Value;
        }

        /// <summary>
        /// Gets the detected CPU features.
        /// </summary>
        /// <returns>A string describing the detected features.</returns>
        public static string GetDetectedFeatures()
        {
            var features = new List<string>();

            if (Sse2.IsSupported) features.Add("SSE2");
            if (Avx.IsSupported) features.Add("AVX");
            if (Avx2.IsSupported) features.Add("AVX2");

            if (Avx512F.IsSupported)
            {
                features.Add("AVX512F");
                // Note: Avx512Vnni is not available in all .NET versions
                // if (Avx512Vnni.IsSupported) features.Add("VNNI");
                if (Avx512BW.IsSupported) features.Add("AVX512BW");
                if (Avx512DQ.IsSupported) features.Add("AVX512DQ");
            }

            return features.Count > 0 ? string.Join(", ", features) : "None";
        }
    }
}
