using System.Collections.Generic;

namespace MLFramework.ModelZoo.Discovery
{
    /// <summary>
    /// Result of a model search query with match scoring.
    /// </summary>
    public class ModelSearchResult
    {
        /// <summary>
        /// Gets or sets the model metadata.
        /// </summary>
        public ModelMetadata Model { get; set; } = null!;

        /// <summary>
        /// Gets or sets the match score (0-1, where 1 is perfect match).
        /// </summary>
        public double MatchScore { get; set; }

        /// <summary>
        /// Gets or sets the reasons why this model matched the query.
        /// </summary>
        public List<string> MatchReasons { get; set; } = new List<string>();

        /// <summary>
        /// Creates a new ModelSearchResult.
        /// </summary>
        public ModelSearchResult()
        {
        }

        /// <summary>
        /// Creates a new ModelSearchResult with the specified model.
        /// </summary>
        /// <param name="model">The model metadata.</param>
        public ModelSearchResult(ModelMetadata model)
        {
            Model = model;
        }

        /// <summary>
        /// Creates a new ModelSearchResult with all properties.
        /// </summary>
        /// <param name="model">The model metadata.</param>
        /// <param name="matchScore">The match score (0-1).</param>
        /// <param name="matchReasons">The reasons why it matched.</param>
        public ModelSearchResult(ModelMetadata model, double matchScore, List<string>? matchReasons = null)
        {
            Model = model;
            MatchScore = matchScore;
            MatchReasons = matchReasons ?? new List<string>();
        }

        /// <summary>
        /// Adds a match reason to the result.
        /// </summary>
        /// <param name="reason">The reason to add.</param>
        public void AddMatchReason(string reason)
        {
            MatchReasons.Add(reason);
        }

        /// <summary>
        /// Returns a string representation of the result.
        /// </summary>
        public override string ToString()
        {
            return $"{Model.Name} (Score: {MatchScore:F2}, {Model.Architecture}, {Model.Task})";
        }
    }
}
