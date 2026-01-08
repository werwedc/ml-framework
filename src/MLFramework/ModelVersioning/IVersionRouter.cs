namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Interface for routing model version requests based on various strategies
    /// </summary>
    public interface IVersionRouter
    {
        /// <summary>
        /// Sets the routing policy to be used for request routing
        /// </summary>
        /// <param name="policy">The routing policy to set</param>
        void SetRoutingPolicy(RoutingPolicy policy);

        /// <summary>
        /// Routes a request to the appropriate model version based on the current policy
        /// </summary>
        /// <param name="context">The context information for the request</param>
        /// <returns>A routing result indicating which version to use and any shadow versions</returns>
        RoutingResult RouteRequest(RequestContext context);

        /// <summary>
        /// Updates the routing policy with a new policy
        /// </summary>
        /// <param name="newPolicy">The new routing policy to apply</param>
        void UpdatePolicy(RoutingPolicy newPolicy);

        /// <summary>
        /// Gets current routing policy
        /// </summary>
        /// <returns>The current routing policy</returns>
        RoutingPolicy GetCurrentPolicy();

        /// <summary>
        /// Gets the default version for a specific model
        /// </summary>
        /// <param name="modelId">The model ID</param>
        /// <returns>The default version for the model</returns>
        string? GetDefaultVersion(string modelId);

        /// <summary>
        /// Sets the default version for a specific model
        /// </summary>
        /// <param name="modelId">The model ID</param>
        /// <param name="version">The version to set as default</param>
        void SetDefaultVersion(string modelId, string version);
    }
}
