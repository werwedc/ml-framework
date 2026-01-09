namespace MLFramework.ModelZoo.Discovery
{
    /// <summary>
    /// Sort criteria for model search results.
    /// </summary>
    public enum SearchSortBy
    {
        /// <summary>
        /// Sort by model name alphabetically.
        /// </summary>
        Name,

        /// <summary>
        /// Sort by accuracy metric (highest accuracy).
        /// </summary>
        Accuracy,

        /// <summary>
        /// Sort by file size (smallest first).
        /// </summary>
        Size,

        /// <summary>
        /// Sort by number of parameters (fewest first).
        /// </summary>
        Parameters,

        /// <summary>
        /// Sort by release date if available (newest first).
        /// </summary>
        Date
    }
}
