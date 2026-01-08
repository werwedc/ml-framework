using System.Text.RegularExpressions;

namespace MLFramework.Serving.Routing;

/// <summary>
/// Represents a semantic version for model versioning
/// </summary>
public readonly struct SemanticVersion : IComparable<SemanticVersion>, IEquatable<SemanticVersion>
{
    public int Major { get; }
    public int Minor { get; }
    public int Patch { get; }
    public string? PreRelease { get; }
    public string? BuildMetadata { get; }

    public SemanticVersion(int major, int minor, int patch, string? preRelease = null, string? buildMetadata = null)
    {
        if (major < 0 || minor < 0 || patch < 0)
            throw new ArgumentException("Version numbers must be non-negative", nameof(major));

        Major = major;
        Minor = minor;
        Patch = patch;
        PreRelease = preRelease;
        BuildMetadata = buildMetadata;
    }

    /// <summary>
    /// Parse a semantic version string (e.g., "1.2.3", "2.0.0-beta.1")
    /// </summary>
    public static SemanticVersion Parse(string version)
    {
        if (string.IsNullOrWhiteSpace(version))
            throw new ArgumentException("Version string cannot be empty", nameof(version));

        // Match semantic version pattern: major.minor.patch[-prerelease][+build]
        var match = Regex.Match(version, @"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-\.]+))?(?:\+([0-9A-Za-z-\.]+))?$");

        if (!match.Success)
            throw new FormatException($"Invalid semantic version format: {version}");

        var major = int.Parse(match.Groups[1].Value);
        var minor = int.Parse(match.Groups[2].Value);
        var patch = int.Parse(match.Groups[3].Value);
        var preRelease = match.Groups[4].Success ? match.Groups[4].Value : null;
        var buildMetadata = match.Groups[5].Success ? match.Groups[5].Value : null;

        return new SemanticVersion(major, minor, patch, preRelease, buildMetadata);
    }

    /// <summary>
    /// Try to parse a semantic version string
    /// </summary>
    public static bool TryParse(string version, out SemanticVersion semanticVersion)
    {
        try
        {
            semanticVersion = Parse(version);
            return true;
        }
        catch
        {
            semanticVersion = default;
            return false;
        }
    }

    public int CompareTo(SemanticVersion other)
    {
        if (Major != other.Major)
            return Major.CompareTo(other.Major);

        if (Minor != other.Minor)
            return Minor.CompareTo(other.Minor);

        if (Patch != other.Patch)
            return Patch.CompareTo(other.Patch);

        // Compare pre-release versions
        return ComparePreRelease(PreRelease, other.PreRelease);
    }

    private static int ComparePreRelease(string? a, string? b)
    {
        // No pre-release is higher than any pre-release
        if (a == null && b == null) return 0;
        if (a == null) return 1;
        if (b == null) return -1;

        // Split identifiers by dots
        var aIdentifiers = a.Split('.');
        var bIdentifiers = b.Split('.');

        var maxLen = Math.Max(aIdentifiers.Length, bIdentifiers.Length);

        for (int i = 0; i < maxLen; i++)
        {
            var aIdent = i < aIdentifiers.Length ? aIdentifiers[i] : null;
            var bIdent = i < bIdentifiers.Length ? bIdentifiers[i] : null;

            // Null identifier is less than non-null
            if (aIdent == null && bIdent == null) continue;
            if (aIdent == null) return -1;
            if (bIdent == null) return 1;

            // Try numeric comparison
            bool aIsNum = int.TryParse(aIdent, out int aNum);
            bool bIsNum = int.TryParse(bIdent, out int bNum);

            if (aIsNum && bIsNum)
            {
                if (aNum != bNum)
                    return aNum.CompareTo(bNum);
            }
            else if (aIsNum)
            {
                // Numeric identifiers have lower precedence than non-numeric
                return -1;
            }
            else if (bIsNum)
            {
                // Numeric identifiers have lower precedence than non-numeric
                return 1;
            }
            else
            {
                // Compare strings lexicographically
                var cmp = string.CompareOrdinal(aIdent, bIdent);
                if (cmp != 0)
                    return cmp;
            }
        }

        return 0;
    }

    public bool Equals(SemanticVersion other)
    {
        return Major == other.Major &&
               Minor == other.Minor &&
               Patch == other.Patch &&
               PreRelease == other.PreRelease;
        // Build metadata is ignored for equality
    }

    public override bool Equals(object? obj)
    {
        return obj is SemanticVersion other && Equals(other);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(Major, Minor, Patch, PreRelease);
    }

    public override string ToString()
    {
        var version = $"{Major}.{Minor}.{Patch}";
        if (!string.IsNullOrEmpty(PreRelease))
            version += $"-{PreRelease}";
        if (!string.IsNullOrEmpty(BuildMetadata))
            version += $"+{BuildMetadata}";
        return version;
    }

    public static bool operator ==(SemanticVersion left, SemanticVersion right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(SemanticVersion left, SemanticVersion right)
    {
        return !left.Equals(right);
    }

    public static bool operator <(SemanticVersion left, SemanticVersion right)
    {
        return left.CompareTo(right) < 0;
    }

    public static bool operator <=(SemanticVersion left, SemanticVersion right)
    {
        return left.CompareTo(right) <= 0;
    }

    public static bool operator >(SemanticVersion left, SemanticVersion right)
    {
        return left.CompareTo(right) > 0;
    }

    public static bool operator >=(SemanticVersion left, SemanticVersion right)
    {
        return left.CompareTo(right) >= 0;
    }
}
