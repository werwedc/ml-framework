using System;

namespace MLFramework.Distributed.Mesh;

/// <summary>
/// Represents a coordinate in a multi-dimensional device mesh.
/// </summary>
public struct DeviceMeshCoord : IEquatable<DeviceMeshCoord>
{
    public readonly int[] Coordinates;

    public DeviceMeshCoord(params int[] coords)
    {
        Coordinates = coords;
    }

    public int this[int dim]
    {
        get => Coordinates[dim];
        set => Coordinates[dim] = value;
    }

    public int Dimensions => Coordinates.Length;

    public override string ToString()
    {
        return $"({string.Join(", ", Coordinates)})";
    }

    public bool Equals(DeviceMeshCoord other)
    {
        if (Dimensions != other.Dimensions)
            return false;
        for (int i = 0; i < Dimensions; i++)
        {
            if (Coordinates[i] != other.Coordinates[i])
                return false;
        }
        return true;
    }

    public override bool Equals(object? obj)
    {
        return obj is DeviceMeshCoord other && Equals(other);
    }

    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            foreach (var coord in Coordinates)
            {
                hash = hash * 31 + coord.GetHashCode();
            }
            return hash;
        }
    }

    public static bool operator ==(DeviceMeshCoord left, DeviceMeshCoord right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(DeviceMeshCoord left, DeviceMeshCoord right)
    {
        return !(left == right);
    }
}
