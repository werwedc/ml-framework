namespace MLFramework.Inference;

/// <summary>
/// Interface for tokenization operations.
/// </summary>
public interface ITokenizer
{
    /// <summary>
    /// Encodes a text prompt into token IDs.
    /// </summary>
    /// <param name="text">The text to encode.</param>
    /// <returns>An array of token IDs.</returns>
    int[] Encode(string text);

    /// <summary>
    /// Decodes token IDs back into text.
    /// </summary>
    /// <param name="tokens">The token IDs to decode.</param>
    /// <returns>The decoded text.</returns>
    string Decode(int[] tokens);
}
