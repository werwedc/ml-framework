using System;
using System.Collections.Generic;

namespace MLFramework.ModelZoo.Discovery
{
    /// <summary>
    /// Query parameters for searching the model registry.
    /// </summary>
    public class ModelSearchQuery
    {
        /// <summary>
        /// Gets or sets the task type filter (optional).
        /// </summary>
        public TaskType? Task { get; set; }

        /// <summary>
        /// Gets or sets the architecture filter (optional).
        /// </summary>
        public string? Architecture { get; set; }

        /// <summary>
        /// Gets or sets the minimum accuracy threshold (optional).
        /// </summary>
        public double? MinAccuracy { get; set; }

        /// <summary>
        /// Gets or sets the maximum accuracy threshold (optional).
        /// </summary>
        public double? MaxAccuracy { get; set; }

        /// <summary>
        /// Gets or sets the maximum file size in bytes (optional).
        /// </summary>
        public long? MaxFileSize { get; set; }

        /// <summary>
        /// Gets or sets the minimum file size in bytes (optional).
        /// </summary>
        public long? MinFileSize { get; set; }

        /// <summary>
        /// Gets or sets the maximum number of parameters (optional).
        /// </summary>
        public long? MaxParameters { get; set; }

        /// <summary>
        /// Gets or sets the minimum number of parameters (optional).
        /// </summary>
        public long? MinParameters { get; set; }

        /// <summary>
        /// Gets or sets the license filter (optional).
        /// </summary>
        public string? License { get; set; }

        /// <summary>
        /// Gets or sets the compatible input shape filter (optional).
        /// </summary>
        public int[]? InputShape { get; set; }

        /// <summary>
        /// Gets or sets the compatible output shape filter (optional).
        /// </summary>
        public int[]? OutputShape { get; set; }

        /// <summary>
        /// Gets or sets the framework version compatibility filter (optional).
        /// </summary>
        public string? FrameworkVersion { get; set; }

        /// <summary>
        /// Gets or sets the pre-trained dataset filter (optional).
        /// </summary>
        public string? PretrainedDataset { get; set; }

        /// <summary>
        /// Gets or sets custom filter functions (optional).
        /// </summary>
        public Dictionary<string, Func<ModelMetadata, bool>>? CustomFilters { get; set; }

        /// <summary>
        /// Gets or sets the sort criteria (default: Name).
        /// </summary>
        public SearchSortBy SortBy { get; set; } = SearchSortBy.Name;

        /// <summary>
        /// Gets or sets whether to sort in descending order (default: true).
        /// </summary>
        public bool SortDescending { get; set; } = true;

        /// <summary>
        /// Gets or sets the maximum number of results (default: 100).
        /// </summary>
        public int Limit { get; set; } = 100;

        /// <summary>
        /// Creates a default search query.
        /// </summary>
        public ModelSearchQuery()
        {
        }

        /// <summary>
        /// Validates the query parameters.
        /// </summary>
        public void Validate()
        {
            if (MinAccuracy.HasValue && MaxAccuracy.HasValue && MinAccuracy > MaxAccuracy)
            {
                throw new ArgumentException("MinAccuracy cannot be greater than MaxAccuracy");
            }

            if (MinFileSize.HasValue && MaxFileSize.HasValue && MinFileSize > MaxFileSize)
            {
                throw new ArgumentException("MinFileSize cannot be greater than MaxFileSize");
            }

            if (MinParameters.HasValue && MaxParameters.HasValue && MinParameters > MaxParameters)
            {
                throw new ArgumentException("MinParameters cannot be greater than MaxParameters");
            }

            if (Limit <= 0)
            {
                throw new ArgumentException("Limit must be positive");
            }
        }
    }
}
