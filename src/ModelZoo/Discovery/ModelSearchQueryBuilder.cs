using System;
using System.Collections.Generic;

namespace MLFramework.ModelZoo.Discovery
{
    /// <summary>
    /// Fluent API for building model search queries.
    /// </summary>
    public class ModelSearchQueryBuilder
    {
        private readonly ModelSearchQuery _query;

        /// <summary>
        /// Creates a new ModelSearchQueryBuilder.
        /// </summary>
        public ModelSearchQueryBuilder()
        {
            _query = new ModelSearchQuery();
        }

        /// <summary>
        /// Sets the task type filter.
        /// </summary>
        /// <param name="task">The task type.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder WithTask(TaskType task)
        {
            _query.Task = task;
            return this;
        }

        /// <summary>
        /// Sets the architecture filter.
        /// </summary>
        /// <param name="architecture">The architecture name.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder WithArchitecture(string architecture)
        {
            _query.Architecture = architecture;
            return this;
        }

        /// <summary>
        /// Sets the accuracy range filter.
        /// </summary>
        /// <param name="min">Minimum accuracy.</param>
        /// <param name="max">Maximum accuracy.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder WithAccuracyRange(double min, double max)
        {
            _query.MinAccuracy = min;
            _query.MaxAccuracy = max;
            return this;
        }

        /// <summary>
        /// Sets the minimum accuracy threshold.
        /// </summary>
        /// <param name="min">Minimum accuracy.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder WithMinAccuracy(double min)
        {
            _query.MinAccuracy = min;
            return this;
        }

        /// <summary>
        /// Sets the maximum accuracy threshold.
        /// </summary>
        /// <param name="max">Maximum accuracy.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder WithMaxAccuracy(double max)
        {
            _query.MaxAccuracy = max;
            return this;
        }

        /// <summary>
        /// Sets the file size range filter.
        /// </summary>
        /// <param name="minBytes">Minimum file size in bytes.</param>
        /// <param name="maxBytes">Maximum file size in bytes.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder WithFileSizeRange(long minBytes, long maxBytes)
        {
            _query.MinFileSize = minBytes;
            _query.MaxFileSize = maxBytes;
            return this;
        }

        /// <summary>
        /// Sets the maximum file size filter.
        /// </summary>
        /// <param name="maxBytes">Maximum file size in bytes.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder WithMaxFileSize(long maxBytes)
        {
            _query.MaxFileSize = maxBytes;
            return this;
        }

        /// <summary>
        /// Sets the parameter count range filter.
        /// </summary>
        /// <param name="minParams">Minimum parameter count.</param>
        /// <param name="maxParams">Maximum parameter count.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder WithParameterRange(long minParams, long maxParams)
        {
            _query.MinParameters = minParams;
            _query.MaxParameters = maxParams;
            return this;
        }

        /// <summary>
        /// Sets the license filter.
        /// </summary>
        /// <param name="license">The license type.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder WithLicense(string license)
        {
            _query.License = license;
            return this;
        }

        /// <summary>
        /// Sets the input shape filter.
        /// </summary>
        /// <param name="shape">The input shape.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder WithInputShape(params int[] shape)
        {
            _query.InputShape = shape;
            return this;
        }

        /// <summary>
        /// Sets the output shape filter.
        /// </summary>
        /// <param name="shape">The output shape.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder WithOutputShape(params int[] shape)
        {
            _query.OutputShape = shape;
            return this;
        }

        /// <summary>
        /// Sets the framework version filter.
        /// </summary>
        /// <param name="version">The framework version.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder WithFrameworkVersion(string version)
        {
            _query.FrameworkVersion = version;
            return this;
        }

        /// <summary>
        /// Sets the pre-trained dataset filter.
        /// </summary>
        /// <param name="dataset">The dataset name.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder WithPretrainedDataset(string dataset)
        {
            _query.PretrainedDataset = dataset;
            return this;
        }

        /// <summary>
        /// Sets the sort criteria.
        /// </summary>
        /// <param name="sortBy">The sort criteria.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder SortBy(SearchSortBy sortBy)
        {
            _query.SortBy = sortBy;
            return this;
        }

        /// <summary>
        /// Sets whether to sort in descending order.
        /// </summary>
        /// <param name="descending">True for descending, false for ascending.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder SortDescending(bool descending)
        {
            _query.SortDescending = descending;
            return this;
        }

        /// <summary>
        /// Sets the result limit.
        /// </summary>
        /// <param name="limit">Maximum number of results.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder WithLimit(int limit)
        {
            _query.Limit = limit;
            return this;
        }

        /// <summary>
        /// Adds a custom filter function.
        /// </summary>
        /// <param name="name">The filter name.</param>
        /// <param name="filter">The filter function.</param>
        /// <returns>The builder for chaining.</returns>
        public ModelSearchQueryBuilder AddCustomFilter(string name, Func<ModelMetadata, bool> filter)
        {
            _query.CustomFilters ??= new Dictionary<string, Func<ModelMetadata, bool>>();
            _query.CustomFilters[name] = filter;
            return this;
        }

        /// <summary>
        /// Builds the query object.
        /// </summary>
        /// <returns>The constructed ModelSearchQuery.</returns>
        public ModelSearchQuery Build()
        {
            _query.Validate();
            return _query;
        }
    }
}
