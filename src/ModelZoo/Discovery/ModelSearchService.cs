using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.ModelZoo.Discovery
{
    /// <summary>
    /// Service for searching models in the model registry.
    /// </summary>
    public class ModelSearchService
    {
        private readonly ModelRegistry _registry;

        /// <summary>
        /// Creates a new ModelSearchService.
        /// </summary>
        /// <param name="registry">The model registry to search.</param>
        public ModelSearchService(ModelRegistry registry)
        {
            _registry = registry ?? throw new ArgumentNullException(nameof(registry));
        }

        /// <summary>
        /// Searches the registry with the specified query.
        /// </summary>
        /// <param name="query">The search query.</param>
        /// <returns>List of search results.</returns>
        public List<ModelSearchResult> Search(ModelSearchQuery query)
        {
            if (query == null)
            {
                throw new ArgumentNullException(nameof(query));
            }

            query.Validate();

            var allModels = _registry.GetAllModels();
            var results = new List<ModelSearchResult>();

            foreach (var model in allModels)
            {
                var (matches, score, reasons) = EvaluateModel(model, query);
                if (matches)
                {
                    var result = new ModelSearchResult(model, score, reasons);
                    results.Add(result);
                }
            }

            // Apply sorting
            SortResults(results, query.SortBy, query.SortDescending);

            // Apply limit
            if (results.Count > query.Limit)
            {
                results = results.Take(query.Limit).ToList();
            }

            return results;
        }

        /// <summary>
        /// Quick search by task type.
        /// </summary>
        /// <param name="task">The task type.</param>
        /// <param name="limit">Maximum number of results.</param>
        /// <returns>List of matching models.</returns>
        public List<ModelSearchResult> SearchByTask(TaskType task, int limit = 10)
        {
            var query = new ModelSearchQuery
            {
                Task = task,
                Limit = limit
            };
            return Search(query);
        }

        /// <summary>
        /// Search by minimum accuracy.
        /// </summary>
        /// <param name="minAccuracy">Minimum accuracy threshold.</param>
        /// <param name="task">Optional task filter.</param>
        /// <param name="limit">Maximum number of results.</param>
        /// <returns>List of matching models.</returns>
        public List<ModelSearchResult> SearchByAccuracy(double minAccuracy, TaskType? task = null, int limit = 10)
        {
            var query = new ModelSearchQuery
            {
                MinAccuracy = minAccuracy,
                Task = task,
                SortBy = SearchSortBy.Accuracy,
                SortDescending = true,
                Limit = limit
            };
            return Search(query);
        }

        /// <summary>
        /// Search by architecture type.
        /// </summary>
        /// <param name="architecture">The architecture name.</param>
        /// <param name="limit">Maximum number of results.</param>
        /// <returns>List of matching models.</returns>
        public List<ModelSearchResult> SearchByArchitecture(string architecture, int limit = 10)
        {
            var query = new ModelSearchQuery
            {
                Architecture = architecture,
                Limit = limit
            };
            return Search(query);
        }

        /// <summary>
        /// Search by maximum file size.
        /// </summary>
        /// <param name="maxSize">Maximum file size in bytes.</param>
        /// <param name="task">Optional task filter.</param>
        /// <param name="limit">Maximum number of results.</param>
        /// <returns>List of matching models.</returns>
        public List<ModelSearchResult> SearchBySize(long maxSize, TaskType? task = null, int limit = 10)
        {
            var query = new ModelSearchQuery
            {
                MaxFileSize = maxSize,
                Task = task,
                SortBy = SearchSortBy.Size,
                SortDescending = false,
                Limit = limit
            };
            return Search(query);
        }

        /// <summary>
        /// Advanced search using a fluent query builder.
        /// </summary>
        /// <param name="build">Action to configure the query builder.</param>
        /// <returns>List of search results.</returns>
        public List<ModelSearchResult> AdvancedSearch(Action<ModelSearchQueryBuilder> build)
        {
            var builder = new ModelSearchQueryBuilder();
            build(builder);
            var query = builder.Build();
            return Search(query);
        }

        /// <summary>
        /// Evaluates a model against a query to determine if it matches and calculate match score.
        /// </summary>
        private (bool matches, double score, List<string> reasons) EvaluateModel(ModelMetadata model, ModelSearchQuery query)
        {
            var matches = true;
            var score = 1.0;
            var reasons = new List<string>();

            // Task filter
            if (query.Task.HasValue && model.Task != query.Task.Value)
            {
                matches = false;
                return (false, 0.0, reasons);
            }

            // Architecture filter (case-insensitive partial match)
            if (!string.IsNullOrEmpty(query.Architecture))
            {
                if (!model.Architecture.Contains(query.Architecture, StringComparison.OrdinalIgnoreCase))
                {
                    matches = false;
                    return (false, 0.0, reasons);
                }
                reasons.Add($"Matches architecture: {model.Architecture}");
            }

            // Accuracy range
            var accuracy = GetPrimaryAccuracy(model);
            if (query.MinAccuracy.HasValue && accuracy < query.MinAccuracy.Value)
            {
                matches = false;
                return (false, 0.0, reasons);
            }
            if (query.MaxAccuracy.HasValue && accuracy > query.MaxAccuracy.Value)
            {
                matches = false;
                return (false, 0.0, reasons);
            }
            if (accuracy > 0)
            {
                reasons.Add($"Accuracy: {accuracy:P2}");
            }

            // File size range
            if (query.MinFileSize.HasValue && model.FileSizeBytes < query.MinFileSize.Value)
            {
                matches = false;
                return (false, 0.0, reasons);
            }
            if (query.MaxFileSize.HasValue && model.FileSizeBytes > query.MaxFileSize.Value)
            {
                matches = false;
                return (false, 0.0, reasons);
            }
            if (model.FileSizeBytes > 0)
            {
                reasons.Add($"Size: {FormatBytes(model.FileSizeBytes)}");
            }

            // Parameter count range
            if (query.MinParameters.HasValue && model.NumParameters < query.MinParameters.Value)
            {
                matches = false;
                return (false, 0.0, reasons);
            }
            if (query.MaxParameters.HasValue && model.NumParameters > query.MaxParameters.Value)
            {
                matches = false;
                return (false, 0.0, reasons);
            }
            if (model.NumParameters > 0)
            {
                reasons.Add($"Parameters: {FormatNumber(model.NumParameters)}");
            }

            // License filter (case-insensitive)
            if (!string.IsNullOrEmpty(query.License))
            {
                if (!model.License.Equals(query.License, StringComparison.OrdinalIgnoreCase))
                {
                    matches = false;
                    return (false, 0.0, reasons);
                }
                reasons.Add($"License: {model.License}");
            }

            // Input shape filter
            if (query.InputShape != null && query.InputShape.Length > 0)
            {
                if (!ShapesMatch(model.InputShape, query.InputShape))
                {
                    matches = false;
                    return (false, 0.0, reasons);
                }
                reasons.Add($"Input shape: [{string.Join(", ", model.InputShape)}]");
            }

            // Output shape filter
            if (query.OutputShape != null && query.OutputShape.Length > 0)
            {
                if (!ShapesMatch(model.OutputShape, query.OutputShape))
                {
                    matches = false;
                    return (false, 0.0, reasons);
                }
                reasons.Add($"Output shape: [{string.Join(", ", model.OutputShape)}]");
            }

            // Framework version filter
            if (!string.IsNullOrEmpty(query.FrameworkVersion))
            {
                // For now, assume match if framework version is specified
                reasons.Add($"Framework: {query.FrameworkVersion}");
            }

            // Pretrained dataset filter (case-insensitive partial match)
            if (!string.IsNullOrEmpty(query.PretrainedDataset))
            {
                if (!model.PretrainedOn.Contains(query.PretrainedDataset, StringComparison.OrdinalIgnoreCase))
                {
                    matches = false;
                    return (false, 0.0, reasons);
                }
                reasons.Add($"Pretrained on: {model.PretrainedOn}");
            }

            // Custom filters
            if (query.CustomFilters != null)
            {
                foreach (var (name, filter) in query.CustomFilters)
                {
                    if (!filter(model))
                    {
                        matches = false;
                        return (false, 0.0, reasons);
                    }
                    reasons.Add($"Custom filter passed: {name}");
                }
            }

            // Calculate match score
            score = CalculateMatchScore(model, query);

            return (matches, score, reasons);
        }

        /// <summary>
        /// Calculates the match score for a model (0-1).
        /// </summary>
        private double CalculateMatchScore(ModelMetadata model, ModelSearchQuery query)
        {
            var totalScore = 1.0;

            // Accuracy score (higher is better)
            if (query.MinAccuracy.HasValue)
            {
                var accuracy = GetPrimaryAccuracy(model);
                var accuracyScore = accuracy / query.MinAccuracy.Value;
                totalScore = Math.Min(totalScore, accuracyScore);
            }

            // Size score (smaller is better if max size is specified)
            if (query.MaxFileSize.HasValue)
            {
                var sizeScore = 1.0 - ((double)model.FileSizeBytes / query.MaxFileSize.Value);
                totalScore = Math.Min(totalScore, sizeScore);
            }

            // Parameter score (fewer is better if max parameters is specified)
            if (query.MaxParameters.HasValue)
            {
                var paramScore = 1.0 - ((double)model.NumParameters / query.MaxParameters.Value);
                totalScore = Math.Min(totalScore, paramScore);
            }

            return Math.Max(0.0, Math.Min(1.0, totalScore));
        }

        /// <summary>
        /// Sorts the search results according to the specified criteria.
        /// </summary>
        private void SortResults(List<ModelSearchResult> results, SearchSortBy sortBy, bool descending)
        {
            IOrderedEnumerable<ModelSearchResult> ordered;

            switch (sortBy)
            {
                case SearchSortBy.Name:
                    ordered = descending
                        ? results.OrderByDescending(r => r.Model.Name)
                        : results.OrderBy(r => r.Model.Name);
                    break;

                case SearchSortBy.Accuracy:
                    ordered = descending
                        ? results.OrderByDescending(r => GetPrimaryAccuracy(r.Model))
                        : results.OrderBy(r => GetPrimaryAccuracy(r.Model));
                    break;

                case SearchSortBy.Size:
                    ordered = descending
                        ? results.OrderByDescending(r => r.Model.FileSizeBytes)
                        : results.OrderBy(r => r.Model.FileSizeBytes);
                    break;

                case SearchSortBy.Parameters:
                    ordered = descending
                        ? results.OrderByDescending(r => r.Model.NumParameters)
                        : results.OrderBy(r => r.Model.NumParameters);
                    break;

                case SearchSortBy.Date:
                    // Sort by version as a proxy for date
                    ordered = descending
                        ? results.OrderByDescending(r => r.Model.Version)
                        : results.OrderBy(r => r.Model.Version);
                    break;

                default:
                    ordered = results.OrderByDescending(r => r.Model.Name);
                    break;
            }

            // Replace the list with sorted results
            var sorted = ordered.ToList();
            results.Clear();
            results.AddRange(sorted);
        }

        /// <summary>
        /// Gets the primary accuracy metric from performance metrics.
        /// </summary>
        private double GetPrimaryAccuracy(ModelMetadata model)
        {
            if (model.PerformanceMetrics.TryGetValue("accuracy", out var accuracy))
            {
                return accuracy;
            }
            if (model.PerformanceMetrics.TryGetValue("top1", out var top1))
            {
                return top1;
            }
            if (model.PerformanceMetrics.TryGetValue("top5", out var top5))
            {
                return top5;
            }
            return 0.0;
        }

        /// <summary>
        /// Checks if two tensor shapes match (exact match or compatible dimensions).
        /// </summary>
        private bool ShapesMatch(int[] shape1, int[] shape2)
        {
            if (shape1.Length != shape2.Length)
            {
                return false;
            }

            for (int i = 0; i < shape1.Length; i++)
            {
                // -1 indicates a dynamic dimension (matches any)
                if (shape1[i] != shape2[i] && shape1[i] != -1 && shape2[i] != -1)
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Formats bytes to human-readable string.
        /// </summary>
        private string FormatBytes(long bytes)
        {
            string[] sizes = { "B", "KB", "MB", "GB", "TB" };
            double len = bytes;
            int order = 0;
            while (len >= 1024 && order < sizes.Length - 1)
            {
                order++;
                len = len / 1024;
            }
            return $"{len:0.##} {sizes[order]}";
        }

        /// <summary>
        /// Formats a number to human-readable string.
        /// </summary>
        private string FormatNumber(long number)
        {
            if (number >= 1_000_000_000)
            {
                return $"{number / 1_000_000_000.0:F1}B";
            }
            if (number >= 1_000_000)
            {
                return $"{number / 1_000_000.0:F1}M";
            }
            if (number >= 1_000)
            {
                return $"{number / 1_000.0:F1}K";
            }
            return number.ToString();
        }
    }
}
