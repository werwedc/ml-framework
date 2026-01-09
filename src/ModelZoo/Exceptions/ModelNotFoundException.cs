using System;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Exception thrown when a requested model is not found in the registry.
    /// </summary>
    public class ModelNotFoundException : Exception
    {
        /// <summary>
        /// Name of the model that was not found.
        /// </summary>
        public string ModelName { get; }

        /// <summary>
        /// Creates a new ModelNotFoundException.
        /// </summary>
        /// <param name="modelName">Name of the model that was not found.</param>
        public ModelNotFoundException(string modelName)
            : base($"Model '{modelName}' not found in the registry.")
        {
            ModelName = modelName;
        }

        /// <summary>
        /// Creates a new ModelNotFoundException with an inner exception.
        /// </summary>
        /// <param name="modelName">Name of the model that was not found.</param>
        /// <param name="innerException">The exception that caused this exception.</param>
        public ModelNotFoundException(string modelName, Exception innerException)
            : base($"Model '{modelName}' not found in the registry.", innerException)
        {
            ModelName = modelName;
        }
    }
}
