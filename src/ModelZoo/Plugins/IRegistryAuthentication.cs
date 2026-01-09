using System.Net.Http;
using System.Net.Http.Headers;

namespace MLFramework.ModelZoo.Plugins
{
    /// <summary>
    /// Base interface for authentication implementations.
    /// </summary>
    public interface IRegistryAuthentication
    {
        /// <summary>
        /// Applies authentication to an HTTP request.
        /// </summary>
        /// <param name="request">The HTTP request to authenticate.</param>
        void Authenticate(HttpRequestMessage request);

        /// <summary>
        /// Gets the authentication scheme name.
        /// </summary>
        string Scheme { get; }
    }

    /// <summary>
    /// API key authentication implementation.
    /// </summary>
    public class ApiKeyAuthentication : IRegistryAuthentication
    {
        private readonly string _apiKey;
        private readonly string _headerName;

        public string Scheme => "ApiKey";

        public ApiKeyAuthentication(string apiKey, string headerName = "X-API-Key")
        {
            _apiKey = apiKey ?? throw new System.ArgumentNullException(nameof(apiKey));
            _headerName = headerName;
        }

        public void Authenticate(HttpRequestMessage request)
        {
            request.Headers.Add(_headerName, _apiKey);
        }
    }

    /// <summary>
    /// OAuth token authentication implementation.
    /// </summary>
    public class TokenAuthentication : IRegistryAuthentication
    {
        private readonly string _token;
        private readonly string _tokenType;

        public string Scheme => "Bearer";

        public TokenAuthentication(string token, string tokenType = "Bearer")
        {
            _token = token ?? throw new System.ArgumentNullException(nameof(token));
            _tokenType = tokenType;
        }

        public void Authenticate(HttpRequestMessage request)
        {
            request.Headers.Authorization = new AuthenticationHeaderValue(_tokenType, _token);
        }
    }

    /// <summary>
    /// Basic authentication implementation (username/password).
    /// </summary>
    public class BasicAuthentication : IRegistryAuthentication
    {
        private readonly string _username;
        private readonly string _password;

        public string Scheme => "Basic";

        public BasicAuthentication(string username, string password)
        {
            _username = username ?? throw new System.ArgumentNullException(nameof(username));
            _password = password ?? throw new System.ArgumentNullException(nameof(password));
        }

        public void Authenticate(HttpRequestMessage request)
        {
            var credentials = System.Convert.ToBase64String(
                System.Text.Encoding.UTF8.GetBytes($"{_username}:{_password}"));
            request.Headers.Authorization = new AuthenticationHeaderValue("Basic", credentials);
        }
    }

    /// <summary>
    /// Custom authentication implementation that allows custom auth logic.
    /// </summary>
    public class CustomAuthentication : IRegistryAuthentication
    {
        private readonly System.Action<HttpRequestMessage> _authAction;
        private readonly string _scheme;

        public string Scheme => _scheme;

        public CustomAuthentication(System.Action<HttpRequestMessage> authAction, string scheme = "Custom")
        {
            _authAction = authAction ?? throw new System.ArgumentNullException(nameof(authAction));
            _scheme = scheme;
        }

        public void Authenticate(HttpRequestMessage request)
        {
            _authAction(request);
        }
    }
}
