namespace MLFramework.Distributed;

/// <summary>
/// High-level API for launching distributed training.
/// </summary>
public static class DistributedLauncher
{
    /// <summary>
    /// Launches distributed training on localhost (multi-GPU on a single machine).
    /// </summary>
    /// <param name="executablePath">The path to the executable to launch.</param>
    /// <param name="numProcesses">The number of processes to launch.</param>
    /// <param name="args">The command-line arguments to pass to each process.</param>
    /// <param name="logDirectory">Optional directory to store process log files (default: "./logs").</param>
    /// <exception cref="DistributedTrainingException">Thrown when a process fails.</exception>
    /// <exception cref="ArgumentNullException">Thrown when executablePath or args is null.</exception>
    /// <exception cref="ArgumentException">Thrown when numProcesses is less than or equal to 0.</exception>
    public static void LaunchLocal(
        string executablePath,
        int numProcesses,
        string[] args,
        string logDirectory = "./logs")
    {
        if (executablePath == null)
        {
            throw new ArgumentNullException(nameof(executablePath));
        }

        if (args == null)
        {
            throw new ArgumentNullException(nameof(args));
        }

        if (numProcesses <= 0)
        {
            throw new ArgumentException("Number of processes must be greater than 0", nameof(numProcesses));
        }

        var launcher = new ProcessLauncher(
            executablePath,
            numProcesses,
            masterAddr: "127.0.0.1",
            masterPort: 29500);

        try
        {
            launcher.Launch(args, logDirectory);
            launcher.WaitForAll();

            if (launcher.HasFailedProcess())
            {
                var (rank, exitCode) = launcher.GetFirstFailedProcess();
                throw new DistributedTrainingException(
                    $"Process rank {rank} failed with exit code {exitCode}",
                    rank,
                    exitCode);
            }
        }
        finally
        {
            launcher.Dispose();
        }
    }

    /// <summary>
    /// Launches distributed training on multiple nodes (multi-node training).
    /// </summary>
    /// <param name="executablePath">The path to the executable to launch.</param>
    /// <param name="numNodes">The number of nodes to use.</param>
    /// <param name="processesPerNode">The number of processes per node.</param>
    /// <param name="args">The command-line arguments to pass to each process.</param>
    /// <param name="masterAddr">The IP address of the master node (rank 0).</param>
    /// <param name="masterPort">The port for process initialization (default: 29500).</param>
    /// <param name="logDirectory">Optional directory to store process log files (default: "./logs").</param>
    /// <exception cref="NotImplementedException">Multi-node launching requires SSH or job scheduler integration.</exception>
    public static void LaunchMultiNode(
        string executablePath,
        int numNodes,
        int processesPerNode,
        string[] args,
        string masterAddr,
        int masterPort = 29500,
        string logDirectory = "./logs")
    {
        // For multi-node, we launch processes on each node
        // This typically requires SSH or a job scheduler (SLURM, etc.)
        // For simplicity, we assume nodes are accessible via SSH

        var totalProcesses = numNodes * processesPerNode;
        var processesPerNodeList = Enumerable.Repeat(processesPerNode, numNodes).ToArray();

        LaunchMultiNodeInternal(
            executablePath,
            processesPerNodeList,
            args,
            masterAddr,
            masterPort,
            logDirectory);
    }

    /// <summary>
    /// Internal method for launching processes on multiple nodes with varying processes per node.
    /// </summary>
    /// <param name="executablePath">The path to the executable to launch.</param>
    /// <param name="processesPerNode">Array specifying the number of processes on each node.</param>
    /// <param name="args">The command-line arguments to pass to each process.</param>
    /// <param name="masterAddr">The IP address of the master node (rank 0).</param>
    /// <param name="masterPort">The port for process initialization.</param>
    /// <param name="logDirectory">Optional directory to store process log files.</param>
    /// <exception cref="NotImplementedException">Multi-node launching requires SSH or job scheduler integration.</exception>
    private static void LaunchMultiNodeInternal(
        string executablePath,
        int[] processesPerNode,
        string[] args,
        string masterAddr,
        int masterPort,
        string logDirectory)
    {
        // This is a placeholder for multi-node implementation
        // In practice, you'd use SSH or job scheduler to launch on remote nodes
        throw new NotImplementedException(
            "Multi-node launching requires SSH or job scheduler integration");
    }
}
