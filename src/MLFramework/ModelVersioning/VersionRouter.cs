using System.Text.RegularExpressions;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Implements request routing based on flexible policies including percentage splits,
    /// shadow testing, deterministic rules, and time-based schedules
    /// </summary>
    public class VersionRouter : IVersionRouter
    {
        private RoutingPolicy _currentPolicy;
        private readonly object _policyLock;
        private readonly Random _hashGenerator;
        private readonly Dictionary<string, string> _defaultVersions;

        /// <summary>
        /// Creates a new VersionRouter instance
        /// </summary>
        public VersionRouter()
        {
            _policyLock = new object();
            _hashGenerator = new Random();
            _defaultVersions = new Dictionary<string, string>();

            // Initialize with a default policy
            _currentPolicy = new RoutingPolicy
            {
                Mode = RoutingMode.Percentage,
                EffectiveDate = DateTime.UtcNow
            };
        }

        /// <summary>
        /// Sets the routing policy to be used for request routing
        /// </summary>
        /// <param name="policy">The routing policy to set</param>
        /// <exception cref="ArgumentException">Thrown when policy validation fails</exception>
        public void SetRoutingPolicy(RoutingPolicy policy)
        {
            if (policy == null)
                throw new ArgumentNullException(nameof(policy));

            ValidatePolicy(policy);

            lock (_policyLock)
            {
                _currentPolicy = policy;
            }
        }

        /// <summary>
        /// Routes a request to the appropriate model version based on the current policy
        /// </summary>
        /// <param name="context">The context information for the request</param>
        /// <returns>A routing result indicating which version to use and any shadow versions</returns>
        /// <exception cref="InvalidOperationException">Thrown when routing fails due to invalid configuration</exception>
        public RoutingResult RouteRequest(RequestContext context)
        {
            if (context == null)
                throw new ArgumentNullException(nameof(context));

            // Capture the current policy to ensure atomicity during routing
            RoutingPolicy policy;
            lock (_policyLock)
            {
                policy = _currentPolicy;
            }

            if (policy.Rules == null || policy.Rules.Count == 0)
                throw new InvalidOperationException("No routing rules defined in the policy");

            return policy.Mode switch
            {
                RoutingMode.Percentage => RoutePercentage(context, policy),
                RoutingMode.Shadow => RouteShadow(context, policy),
                RoutingMode.Deterministic => RouteDeterministic(context, policy),
                RoutingMode.TimeBased => RouteTimeBased(context, policy),
                _ => throw new NotSupportedException($"Routing mode {policy.Mode} is not supported")
            };
        }

        /// <summary>
        /// Updates the routing policy with a new policy
        /// </summary>
        /// <param name="newPolicy">The new routing policy to apply</param>
        /// <exception cref="ArgumentException">Thrown when policy validation fails</exception>
        public void UpdatePolicy(RoutingPolicy newPolicy)
        {
            if (newPolicy == null)
                throw new ArgumentNullException(nameof(newPolicy));

            ValidatePolicy(newPolicy);

            lock (_policyLock)
            {
                _currentPolicy = newPolicy;
            }
        }

        /// <summary>
        /// Gets the current routing policy
        /// </summary>
        /// <returns>The current routing policy</returns>
        public RoutingPolicy GetCurrentPolicy()
        {
            lock (_policyLock)
            {
                return _currentPolicy;
            }
        }

        #region Routing Modes

        /// <summary>
        /// Routes based on percentage split across versions
        /// </summary>
        private RoutingResult RoutePercentage(RequestContext context, RoutingPolicy policy)
        {
            // Use deterministic hashing for consistent routing of same user
            double hashValue;
            if (!string.IsNullOrEmpty(context.UserId))
            {
                hashValue = GetDeterministicHash(context.UserId);
            }
            else
            {
                // Fall back to random if no user ID
                lock (_hashGenerator)
                {
                    hashValue = _hashGenerator.NextDouble() * 100;
                }
            }

            double cumulativePercentage = 0;
            foreach (var rule in policy.Rules)
            {
                cumulativePercentage += rule.Percentage;
                if (hashValue < cumulativePercentage)
                {
                    return new RoutingResult
                    {
                        Version = rule.Version,
                        IsShadow = false,
                        ShadowVersions = new List<string>(),
                        RuleMatched = $"Percentage({rule.Percentage}%): {rule.Version}"
                    };
                }
            }

            // Default to first rule if no match
            var defaultRule = policy.Rules[0];
            return new RoutingResult
            {
                Version = defaultRule.Version,
                IsShadow = false,
                ShadowVersions = new List<string>(),
                RuleMatched = "Default: " + defaultRule.Version
            };
        }

        /// <summary>
        /// Routes to multiple versions for shadow testing
        /// </summary>
        private RoutingResult RouteShadow(RequestContext context, RoutingPolicy policy)
        {
            var primaryRules = policy.Rules.Where(r => r.IsPrimary).ToList();
            if (primaryRules.Count == 0)
                throw new InvalidOperationException("No primary version defined for shadow routing");

            var primaryRule = primaryRules[0];
            var shadowVersions = policy.Rules
                .Where(r => !r.IsPrimary && r.Version != primaryRule.Version)
                .Select(r => r.Version!)
                .Where(v => v != null)
                .ToList();

            return new RoutingResult
            {
                Version = primaryRule.Version,
                IsShadow = false,
                ShadowVersions = shadowVersions,
                RuleMatched = $"Shadow mode: Primary {primaryRule.Version}, Shadows: {string.Join(", ", shadowVersions)}"
            };
        }

        /// <summary>
        /// Routes based on deterministic rules (user ID pattern, segment, region)
        /// </summary>
        private RoutingResult RouteDeterministic(RequestContext context, RoutingPolicy policy)
        {
            foreach (var rule in policy.Rules)
            {
                // Check user ID pattern match
                if (!string.IsNullOrEmpty(rule.UserIdPattern) &&
                    !string.IsNullOrEmpty(context.UserId))
                {
                    try
                    {
                        if (Regex.IsMatch(context.UserId, rule.UserIdPattern))
                        {
                            return new RoutingResult
                            {
                                Version = rule.Version,
                                IsShadow = false,
                                ShadowVersions = new List<string>(),
                                RuleMatched = $"UserPattern: {rule.UserIdPattern}"
                            };
                        }
                    }
                    catch (ArgumentException ex)
                    {
                        throw new InvalidOperationException($"Invalid user ID pattern: {rule.UserIdPattern}", ex);
                    }
                }

                // Check segment match
                if (!string.IsNullOrEmpty(rule.Segment) &&
                    !string.IsNullOrEmpty(context.Segment) &&
                    rule.Segment.Equals(context.Segment, StringComparison.OrdinalIgnoreCase))
                {
                    return new RoutingResult
                    {
                        Version = rule.Version,
                        IsShadow = false,
                        ShadowVersions = new List<string>(),
                        RuleMatched = $"Segment: {rule.Segment}"
                    };
                }

                // Check region match
                if (!string.IsNullOrEmpty(rule.Region) &&
                    !string.IsNullOrEmpty(context.Region) &&
                    rule.Region.Equals(context.Region, StringComparison.OrdinalIgnoreCase))
                {
                    return new RoutingResult
                    {
                        Version = rule.Version,
                        IsShadow = false,
                        ShadowVersions = new List<string>(),
                        RuleMatched = $"Region: {rule.Region}"
                    };
                }
            }

            // Default to first rule if no match
            var defaultRule = policy.Rules[0];
            return new RoutingResult
            {
                Version = defaultRule.Version,
                IsShadow = false,
                ShadowVersions = new List<string>(),
                RuleMatched = "Default: " + defaultRule.Version
            };
        }

        /// <summary>
        /// Routes based on time-based schedules
        /// </summary>
        private RoutingResult RouteTimeBased(RequestContext context, RoutingPolicy policy)
        {
            foreach (var rule in policy.Rules)
            {
                if (rule.TimeRange != null && rule.TimeRange.IsInRange(context.RequestTime))
                {
                    return new RoutingResult
                    {
                        Version = rule.Version,
                        IsShadow = false,
                        ShadowVersions = new List<string>(),
                        RuleMatched = $"TimeRange: {rule.TimeRange}"
                    };
                }
            }

            // Default to first rule if no match
            var defaultRule = policy.Rules[0];
            return new RoutingResult
            {
                Version = defaultRule.Version,
                IsShadow = false,
                ShadowVersions = new List<string>(),
                RuleMatched = "Default: " + defaultRule.Version
            };
        }

        #endregion

        #region Policy Validation

        /// <summary>
        /// Validates that the routing policy is correctly configured
        /// </summary>
        /// <param name="policy">The policy to validate</param>
        /// <exception cref="ArgumentException">Thrown when validation fails</exception>
        private void ValidatePolicy(RoutingPolicy policy)
        {
            if (policy.Rules == null || policy.Rules.Count == 0)
                throw new ArgumentException("Policy must have at least one rule");

            // Mode-specific validation
            switch (policy.Mode)
            {
                case RoutingMode.Percentage:
                    ValidatePercentagePolicy(policy);
                    break;

                case RoutingMode.Shadow:
                    ValidateShadowPolicy(policy);
                    break;

                case RoutingMode.Deterministic:
                    ValidateDeterministicPolicy(policy);
                    break;

                case RoutingMode.TimeBased:
                    ValidateTimeBasedPolicy(policy);
                    break;
            }

            // Validate that all rules have valid versions
            foreach (var rule in policy.Rules)
            {
                if (string.IsNullOrEmpty(rule.Version))
                    throw new ArgumentException("All rules must have a valid version");
            }
        }

        /// <summary>
        /// Validates percentage-based routing policy
        /// </summary>
        private void ValidatePercentagePolicy(RoutingPolicy policy)
        {
            double totalPercentage = policy.Rules.Sum(r => r.Percentage);

            if (Math.Abs(totalPercentage - 100.0) > 0.01)
                throw new ArgumentException(
                    $"Total percentage must equal 100 (current: {totalPercentage:F2}%)");

            foreach (var rule in policy.Rules)
            {
                if (rule.Percentage < 0 || rule.Percentage > 100)
                    throw new ArgumentException(
                        $"Percentage must be between 0 and 100 (rule {rule.Version}: {rule.Percentage})");
            }
        }

        /// <summary>
        /// Validates shadow mode routing policy
        /// </summary>
        private void ValidateShadowPolicy(RoutingPolicy policy)
        {
            var primaryRules = policy.Rules.Where(r => r.IsPrimary).ToList();
            if (primaryRules.Count == 0)
                throw new ArgumentException("Shadow mode requires at least one primary version");

            if (primaryRules.Count > 1)
                throw new ArgumentException("Shadow mode can only have one primary version");
        }

        /// <summary>
        /// Validates deterministic routing policy
        /// </summary>
        private void ValidateDeterministicPolicy(RoutingPolicy policy)
        {
            bool hasCriteria = policy.Rules.Any(r =>
                !string.IsNullOrEmpty(r.UserIdPattern) ||
                !string.IsNullOrEmpty(r.Segment) ||
                !string.IsNullOrEmpty(r.Region));

            if (!hasCriteria)
                throw new ArgumentException(
                    "Deterministic mode requires at least one rule with user ID pattern, segment, or region");
        }

        /// <summary>
        /// Validates time-based routing policy
        /// </summary>
        private void ValidateTimeBasedPolicy(RoutingPolicy policy)
        {
            bool hasValidTimeRange = policy.Rules.Any(r =>
                r.TimeRange != null && r.TimeRange.IsValid());

            if (!hasValidTimeRange)
                throw new ArgumentException(
                    "TimeBased mode requires at least one rule with a valid time range");
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Generates a deterministic hash value from a string for consistent routing
        /// </summary>
        /// <param name="input">The input string to hash</param>
        /// <returns>A hash value between 0 and 100</returns>
        private double GetDeterministicHash(string input)
        {
            if (string.IsNullOrEmpty(input))
                return 0;

            // Use a simple hash algorithm for deterministic results
            // In production, consider using a more robust hash like MurmurHash3
            var hash = 0;
            foreach (var c in input)
            {
                hash = (hash * 31 + c) % int.MaxValue;
            }

            // Convert to 0-100 range
            return (Math.Abs(hash) % 10000) / 100.0;
        }

        /// <summary>
        /// Gets the default version for a specific model
        /// </summary>
        /// <param name="modelId">The model ID</param>
        /// <returns>The default version for the model, or null if not set</returns>
        public string? GetDefaultVersion(string modelId)
        {
            if (string.IsNullOrEmpty(modelId))
                throw new ArgumentNullException(nameof(modelId));

            lock (_defaultVersions)
            {
                return _defaultVersions.TryGetValue(modelId, out var version) ? version : null;
            }
        }

        /// <summary>
        /// Sets the default version for a specific model
        /// </summary>
        /// <param name="modelId">The model ID</param>
        /// <param name="version">The version to set as default</param>
        public void SetDefaultVersion(string modelId, string version)
        {
            if (string.IsNullOrEmpty(modelId))
                throw new ArgumentNullException(nameof(modelId));

            if (string.IsNullOrEmpty(version))
                throw new ArgumentNullException(nameof(version));

            lock (_defaultVersions)
            {
                _defaultVersions[modelId] = version;
            }
        }

        #endregion
    }
}
