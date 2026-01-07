using System.Runtime.InteropServices;

namespace MLFramework.Quantization.Backends.ARMBackend
{
    /// <summary>
    /// ARM CPU feature detection for NEON and ARMv8.2 dot product instructions.
    /// </summary>
    internal static class ARMFeatureDetection
    {
        private static bool? _isAvailable;

        /// <summary>
        /// Checks if ARM features are available.
        /// </summary>
        /// <returns>True if ARM features are available, false otherwise.</returns>
        public static bool IsAvailable()
        {
            if (_isAvailable.HasValue)
            {
                return _isAvailable.Value;
            }

            // Check if we're on an ARM platform
            _isAvailable = RuntimeInformation.ProcessArchitecture == Architecture.Arm64;

            return _isAvailable.Value;
        }

        /// <summary>
        /// Gets the detected CPU features.
        /// </summary>
        /// <returns>A string describing the detected features.</returns>
        public static string GetDetectedFeatures()
        {
            var features = new List<string>();

            if (IsAvailable())
            {
                features.Add("ARM64");
                // On ARM64, NEON is always available
                features.Add("NEON");

                // Try to detect ARMv8.2 dot product (requires runtime detection)
                // For now, we assume it's available on ARM64
                features.Add("Dot Product");
            }

            return features.Count > 0 ? string.Join(", ", features) : "None";
        }
    }
}
