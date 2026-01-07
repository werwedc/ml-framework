using System;
using System.Threading.Tasks;

namespace MLFramework.Distributed.Mesh;

/// <summary>
/// Static helper methods for working with device mesh state.
/// </summary>
public static class MeshState
{
    /// <summary>
    /// Get the TP world size from mesh.
    /// </summary>
    public static int GetTPWorldSize(DeviceMesh mesh)
    {
        if (mesh == null)
            throw new ArgumentNullException(nameof(mesh));

        return mesh.Shape[(int)ParallelismDimension.Tensor];
    }

    /// <summary>
    /// Get the DP world size from mesh.
    /// </summary>
    public static int GetDPWorldSize(DeviceMesh mesh)
    {
        if (mesh == null)
            throw new ArgumentNullException(nameof(mesh));

        return mesh.Shape[(int)ParallelismDimension.Data];
    }

    /// <summary>
    /// Check if this is the master rank (0, 0, ...).
    /// </summary>
    public static bool IsMasterRank(DeviceMesh mesh)
    {
        if (mesh == null)
            throw new ArgumentNullException(nameof(mesh));

        foreach (var coord in mesh.MyCoordinate.Coordinates)
        {
            if (coord != 0)
                return false;
        }
        return true;
    }

    /// <summary>
    /// Execute code only on master rank (0, 0, ...).
    /// </summary>
    public static Task ExecuteOnMasterAsync(DeviceMesh mesh, Func<Task> action)
    {
        if (mesh == null)
            throw new ArgumentNullException(nameof(mesh));

        if (action == null)
            throw new ArgumentNullException(nameof(action));

        if (IsMasterRank(mesh))
        {
            return action();
        }
        return Task.CompletedTask;
    }

    /// <summary>
    /// Execute synchronous code only on master rank (0, 0, ...).
    /// </summary>
    public static void ExecuteOnMaster(DeviceMesh mesh, Action action)
    {
        if (mesh == null)
            throw new ArgumentNullException(nameof(mesh));

        if (action == null)
            throw new ArgumentNullException(nameof(action));

        if (IsMasterRank(mesh))
        {
            action();
        }
    }
}
