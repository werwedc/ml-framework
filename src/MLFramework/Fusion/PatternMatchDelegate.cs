namespace MLFramework.Fusion;

/// <summary>
/// Delegate for pattern matching
/// </summary>
/// <param name="operations">Operations to match</param>
/// <returns>True if the pattern matches</returns>
public delegate bool PatternMatchDelegate(IReadOnlyList<Operation> operations);
