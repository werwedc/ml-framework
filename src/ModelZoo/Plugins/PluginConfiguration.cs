using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.ModelZoo.Plugins
{
    /// <summary>
    /// Base class for plugin configuration.
    /// Mark configuration classes with [RegistryPluginConfig] attribute for auto-discovery.
    /// </summary>
    public abstract class PluginConfigurationBase
    {
        /// <summary>
        /// Gets or sets the registry URL.
        /// </summary>
        public string RegistryUrl { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets whether caching is enabled.
        /// </summary>
        public bool EnableCaching { get; set; } = true;

        /// <summary>
        /// Gets or sets the cache directory path.
        /// </summary>
        public string CacheDirectory { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the cache expiration in hours.
        /// </summary>
        public int CacheExpirationHours { get; set; } = 24;

        /// <summary>
        /// Gets or sets the HTTP timeout in seconds.
        /// </summary>
        public int HttpTimeoutSeconds { get; set; } = 300;

        /// <summary>
        /// Gets or sets the maximum number of retries.
        /// </summary>
        public int MaxRetries { get; set; } = 3;

        /// <summary>
        /// Gets or sets the retry delay in milliseconds.
        /// </summary>
        public int RetryDelayMilliseconds { get; set; } = 1000;

        /// <summary>
        /// Gets or sets custom configuration properties.
        /// </summary>
        public Dictionary<string, string> CustomProperties { get; set; } = new Dictionary<string, string>();

        /// <summary>
        /// Gets a custom property value.
        /// </summary>
        /// <param name="key">The property key.</param>
        /// <returns>The property value, or null if not found.</returns>
        public string GetCustomProperty(string key)
        {
            CustomProperties.TryGetValue(key, out var value);
            return value;
        }

        /// <summary>
        /// Sets a custom property value.
        /// </summary>
        /// <param name="key">The property key.</param>
        /// <param name="value">The property value.</param>
        public void SetCustomProperty(string key, string value)
        {
            CustomProperties[key] = value;
        }
    }

    /// <summary>
    /// Configuration for authenticated plugins.
    /// </summary>
    public class AuthenticatedPluginConfiguration : PluginConfigurationBase
    {
        /// <summary>
        /// Gets or sets the authentication type (ApiKey, Token, Basic, Custom).
        /// </summary>
        public string AuthenticationType { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the API key (for ApiKey authentication).
        /// </summary>
        public string ApiKey { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the API key header name (default: X-API-Key).
        /// </summary>
        public string ApiKeyHeaderName { get; set; } = "X-API-Key";

        /// <summary>
        /// Gets or sets the OAuth token (for Token authentication).
        /// </summary>
        public string Token { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the token type (default: Bearer).
        /// </summary>
        public string TokenType { get; set; } = "Bearer";

        /// <summary>
        /// Gets or sets the username (for Basic authentication).
        /// </summary>
        public string Username { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the password (for Basic authentication).
        /// </summary>
        public string Password { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets whether credentials are encrypted.
        /// </summary>
        public bool CredentialsEncrypted { get; set; } = false;
    }

    /// <summary>
    /// Helper class for loading plugin configurations.
    /// </summary>
    public static class PluginConfigurationLoader
    {
        /// <summary>
        /// Loads configuration from an environment variable.
        /// </summary>
        /// <param name="envVarName">The environment variable name.</param>
        /// <returns>The configuration value, or empty string if not found.</returns>
        public static string LoadFromEnvironment(string envVarName)
        {
            var value = Environment.GetEnvironmentVariable(envVarName);
            return value ?? string.Empty;
        }

        /// <summary>
        /// Loads configuration from environment variables with a prefix.
        /// </summary>
        /// <typeparam name="T">The configuration type.</typeparam>
        /// <param name="prefix">The environment variable prefix.</param>
        /// <returns>A new instance of the configuration type.</returns>
        public static T LoadFromEnvironment<T>(string prefix) where T : PluginConfigurationBase, new()
        {
            var config = new T();
            var properties = typeof(T).GetProperties(
                System.Reflection.BindingFlags.Public |
                System.Reflection.BindingFlags.Instance |
                System.Reflection.BindingFlags.FlattenHierarchy);

            foreach (var prop in properties)
            {
                if (prop.PropertyType == typeof(string))
                {
                    var envVarName = $"{prefix}_{prop.Name}";
                    var value = LoadFromEnvironment(envVarName);
                    if (!string.IsNullOrEmpty(value))
                    {
                        prop.SetValue(config, value);
                    }
                }
                else if (prop.PropertyType == typeof(bool))
                {
                    var envVarName = $"{prefix}_{prop.Name}";
                    var value = LoadFromEnvironment(envVarName);
                    if (!string.IsNullOrEmpty(value) && bool.TryParse(value, out var boolValue))
                    {
                        prop.SetValue(config, boolValue);
                    }
                }
                else if (prop.PropertyType == typeof(int))
                {
                    var envVarName = $"{prefix}_{prop.Name}";
                    var value = LoadFromEnvironment(envVarName);
                    if (!string.IsNullOrEmpty(value) && int.TryParse(value, out var intValue))
                    {
                        prop.SetValue(config, intValue);
                    }
                }
            }

            return config;
        }
    }
}
