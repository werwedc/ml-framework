using System;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Exception thrown when a model's architecture does not match the expected structure.
    /// </summary>
    public class IncompatibleModelException : Exception
    {
        /// <summary>
        /// Name of the model.
        /// </summary>
        public string ModelName { get; }

        /// <summary>
        /// Version of the model.
        /// </summary>
        public string Version { get; }

        /// <summary>
        /// Expected architecture type.
        /// </summary>
        public string ExpectedArchitecture { get; }

        /// <summary>
        /// Actual architecture type found in the model file.
        /// </summary>
        public string ActualArchitecture { get; }

        /// <summary>
        /// Creates a new IncompatibleModelException.
        /// </summary>
        /// <param name="modelName">Name of the model.</param>
        /// <param name="version">Version of the model.</param>
        /// <param name="expectedArchitecture">Expected architecture type.</param>
        /// <param name="actualArchitecture">Actual architecture type found.</param>
        public IncompatibleModelException(string modelName, string version, string expectedArchitecture, string actualArchitecture)
            : base($"Model '{modelName}' (v{version}) is incompatible. Expected architecture: {expectedArchitecture}, Actual: {actualArchitecture}")
        {
            ModelName = modelName;
            Version = version;
            ExpectedArchitecture = expectedArchitecture;
            ActualArchitecture = actualArchitecture;
        }

        /// <summary>
        /// Creates a new IncompatibleModelException with an inner exception.
        /// </summary>
        /// <param name="modelName">Name of the model.</param>
        /// <param name="version">Version of the model.</param>
        /// <param name="expectedArchitecture">Expected architecture type.</param>
        /// <param name="actualArchitecture">Actual architecture type found.</param>
        /// <param name="innerException">The exception that caused this exception.</param>
        public IncompatibleModelException(string modelName, string version, string expectedArchitecture, string actualArchitecture, Exception innerException)
            : base($"Model '{modelName}' (v{version}) is incompatible. Expected architecture: {expectedArchitecture}, Actual: {actualArchitecture}", innerException)
        {
            ModelName = modelName;
            Version = version;
            ExpectedArchitecture = expectedArchitecture;
            ActualArchitecture = actualArchitecture;
        }
    }
}
