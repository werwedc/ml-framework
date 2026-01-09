namespace MLFramework.ModelZoo.ExternalHubs;

/// <summary>
/// Base interface for hub authentication methods.
/// </summary>
public interface IHubAuthentication
{
    /// <summary>
    /// Gets the type of authentication.
    /// </summary>
    string AuthType { get; }

    /// <summary>
    /// Gets the authentication header value for HTTP requests.
    /// </summary>
    /// <returns>A tuple containing the header name and value.</returns>
    (string HeaderName, string HeaderValue) GetAuthHeader();

    /// <summary>
    /// Validates that the authentication is properly configured.
    /// </summary>
    /// <returns>True if the authentication is valid, false otherwise.</returns>
    bool IsValid();
}

/// <summary>
/// No authentication required (anonymous access).
/// </summary>
public class AnonymousAuth : IHubAuthentication
{
    /// <summary>
    /// Gets the authentication type.
    /// </summary>
    public string AuthType => "anonymous";

    /// <summary>
    /// Gets the authentication header (empty for anonymous access).
    /// </summary>
    /// <returns>An empty tuple indicating no authentication header.</returns>
    public (string HeaderName, string HeaderValue) GetAuthHeader()
    {
        return (string.Empty, string.Empty);
    }

    /// <summary>
    /// Validates that the authentication is properly configured.
    /// </summary>
    /// <returns>Always returns true for anonymous authentication.</returns>
    public bool IsValid()
    {
        return true;
    }
}

/// <summary>
/// API key-based authentication.
/// </summary>
public class ApiKeyAuth : IHubAuthentication
{
    private readonly string _apiKey;

    /// <summary>
    /// Gets the authentication type.
    /// </summary>
    public string AuthType => "api_key";

    /// <summary>
    /// Gets the header name for the API key.
    /// </summary>
    public string HeaderName { get; }

    /// <summary>
    /// Initializes a new instance of the ApiKeyAuth class.
    /// </summary>
    /// <param name="apiKey">The API key.</param>
    /// <param name="headerName">The header name (default: "Authorization").</param>
    public ApiKeyAuth(string apiKey, string headerName = "Authorization")
    {
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            throw new ArgumentException("API key cannot be null or empty.", nameof(apiKey));
        }

        _apiKey = apiKey;
        HeaderName = headerName ?? "Authorization";
    }

    /// <summary>
    /// Gets the authentication header value for HTTP requests.
    /// </summary>
    /// <returns>A tuple containing the header name and API key value.</returns>
    public (string HeaderName, string HeaderValue) GetAuthHeader()
    {
        return (HeaderName, $"Bearer {_apiKey}");
    }

    /// <summary>
    /// Validates that the authentication is properly configured.
    /// </summary>
    /// <returns>True if the API key is not empty, false otherwise.</returns>
    public bool IsValid()
    {
        return !string.IsNullOrWhiteSpace(_apiKey);
    }
}

/// <summary>
/// OAuth token-based authentication.
/// </summary>
public class TokenAuth : IHubAuthentication
{
    private readonly string _token;

    /// <summary>
    /// Gets the authentication type.
    /// </summary>
    public string AuthType => "token";

    /// <summary>
    /// Gets the header name for the token.
    /// </summary>
    public string HeaderName { get; }

    /// <summary>
    /// Gets the token type (e.g., "Bearer", "Token").
    /// </summary>
    public string TokenType { get; }

    /// <summary>
    /// Initializes a new instance of the TokenAuth class.
    /// </summary>
    /// <param name="token">The OAuth token.</param>
    /// <param name="tokenType">The token type (default: "Bearer").</param>
    /// <param name="headerName">The header name (default: "Authorization").</param>
    public TokenAuth(string token, string tokenType = "Bearer", string headerName = "Authorization")
    {
        if (string.IsNullOrWhiteSpace(token))
        {
            throw new ArgumentException("Token cannot be null or empty.", nameof(token));
        }

        _token = token;
        TokenType = tokenType ?? "Bearer";
        HeaderName = headerName ?? "Authorization";
    }

    /// <summary>
    /// Gets the authentication header value for HTTP requests.
    /// </summary>
    /// <returns>A tuple containing the header name and token value.</returns>
    public (string HeaderName, string HeaderValue) GetAuthHeader()
    {
        return (HeaderName, $"{TokenType} {_token}");
    }

    /// <summary>
    /// Validates that the authentication is properly configured.
    /// </summary>
    /// <returns>True if the token is not empty, false otherwise.</returns>
    public bool IsValid()
    {
        return !string.IsNullOrWhiteSpace(_token);
    }
}
