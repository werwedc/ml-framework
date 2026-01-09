using System;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Exception thrown when a requested model version is not found in the registry.
    /// </summary>
    public class VersionNotFoundException : Exception
    {
        /// <summary>
        /// Name of the model.
        /// </summary>
        public string ModelName { get; }

        /// <summary>
        /// Version that was not found.
        /// </summary>
        public string Version { get; }

        /// <summary>
        /// Creates a new VersionNotFoundException.
        /// </summary>
        /// <param name="modelName">Name of the model.</param>
        /// <param name="version">Version that was not found.</param>
        public VersionNotFoundException(string modelName, string version)
            : base($"Version '{version}' not found for model '{modelName}'.")
        {
            ModelName = modelName;
            Version = version;
        }

        /// <summary>
        /// Creates a new VersionNotFoundException with an inner exception.
        /// </summary>
        /// <param name="modelName">Name of the model.</param>
        /// <param name="version">Version that was not found.</param>
        /// <param name="innerException">The exception that caused this exception.</param>
        public VersionNotFoundException(string modelName, string version, Exception innerException)
            : base($"Version '{version}' not found for model '{modelName}'.", innerException)
        {
            ModelName = modelName;
            Version = version;
        }
    }
}
