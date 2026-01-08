namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Represents the result of a health check for a model version.
    /// </summary>
    public class HealthCheckResult
    {
        /// <summary>
        /// Gets or sets a value indicating whether the model version is healthy.
        /// </summary>
        public bool IsHealthy { get; set; }

        /// <summary>
        /// Gets or sets a message describing the health check result.
        /// </summary>
        public string Message { get; set; }

        /// <summary>
        /// Gets or sets the timestamp when the health check was performed.
        /// </summary>
        public DateTime CheckTimestamp { get; set; }

        /// <summary>
        /// Gets or sets diagnostic information about the health check.
        /// </summary>
        public Dictionary<string, object> Diagnostics { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="HealthCheckResult"/> class.
        /// </summary>
        public HealthCheckResult()
        {
            Diagnostics = new Dictionary<string, object>();
            CheckTimestamp = DateTime.UtcNow;
        }

        /// <summary>
        /// Creates a healthy health check result.
        /// </summary>
        /// <param name="message">Optional message.</param>
        /// <returns>A healthy health check result.</returns>
        public static HealthCheckResult Healthy(string? message = null)
        {
            return new HealthCheckResult
            {
                IsHealthy = true,
                Message = message ?? "Model is healthy",
                CheckTimestamp = DateTime.UtcNow
            };
        }

        /// <summary>
        /// Creates an unhealthy health check result.
        /// </summary>
        /// <param name="message">The failure message.</param>
        /// <returns>An unhealthy health check result.</returns>
        public static HealthCheckResult Unhealthy(string message)
        {
            return new HealthCheckResult
            {
                IsHealthy = false,
                Message = message,
                CheckTimestamp = DateTime.UtcNow
            };
        }
    }
}
