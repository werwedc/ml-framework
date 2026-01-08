namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Metadata associated with a model version.
    /// </summary>
    public class ModelMetadata
    {
        /// <summary>
        /// Gets or sets the name of the model.
        /// </summary>
        public string ModelName { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the description of the model.
        /// </summary>
        public string? Description { get; set; }

        /// <summary>
        /// Gets or sets the model framework (e.g., PyTorch, TensorFlow).
        /// </summary>
        public string Framework { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the model architecture or type.
        /// </summary>
        public string Architecture { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the model's input tensor shape.
        /// </summary>
        public int[]? InputShape { get; set; }

        /// <summary>
        /// Gets or sets the model's output tensor shape.
        /// </summary>
        public int[]? OutputShape { get; set; }

        /// <summary>
        /// Gets or sets any additional custom metadata.
        /// </summary>
        public Dictionary<string, string>? CustomMetadata { get; set; }
    }
}
