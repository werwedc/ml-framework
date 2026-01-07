# Spec: Process Launcher

## Overview
Implement a mechanism to launch multiple processes for distributed training, with each process handling one GPU. This provides a convenient way to start distributed training without manually managing processes.

## Requirements
- Launch multiple processes, one per GPU
- Set up environment variables for each process (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
- Support launching on localhost (multi-GPU on single machine)
- Support launching on multiple machines (multi-node training)
- Handle process lifecycle (start, wait, terminate)
- Provide status reporting for all processes

## Classes

### 1. ProcessLauncher Class
```csharp
public class ProcessLauncher : IDisposable
{
    private readonly string _executablePath;
    private readonly int _numProcesses;
    private readonly string _masterAddr;
    private readonly int _masterPort;
    private readonly Process[] _processes;
    private readonly StreamWriter[] _logWriters;

    public ProcessLauncher(
        string executablePath,
        int numProcesses,
        string masterAddr = "127.0.0.1",
        int masterPort = 29500)
    {
        _executablePath = executablePath;
        _numProcesses = numProcesses;
        _masterAddr = masterAddr;
        _masterPort = masterPort;
        _processes = new Process[numProcesses];
        _logWriters = new StreamWriter[numProcesses];
    }

    /// <summary>
    /// Launch all processes.
    /// </summary>
    public void Launch(string[] args, string logDirectory = null)
    {
        for (int rank = 0; rank < _numProcesses; rank++)
        {
            LaunchProcess(rank, args, logDirectory);
        }
    }

    /// <summary>
    /// Launch a specific process by rank.
    /// </summary>
    private void LaunchProcess(int rank, string[] args, string logDirectory)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = _executablePath,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        // Set environment variables
        startInfo.Environment["RANK"] = rank.ToString();
        startInfo.Environment["WORLD_SIZE"] = _numProcesses.ToString();
        startInfo.Environment["MASTER_ADDR"] = _masterAddr;
        startInfo.Environment["MASTER_PORT"] = _masterPort.ToString();

        // Set CUDA device for this rank
        startInfo.Environment["CUDA_VISIBLE_DEVICES"] = rank.ToString();

        // Build arguments
        var allArgs = new List<string>(args);
        allArgs.Add($"--distributed-rank={rank}");
        allArgs.Add($"--distributed-world-size={_numProcesses}");
        startInfo.Arguments = string.Join(" ", allArgs);

        // Create process
        var process = new Process { StartInfo = startInfo };
        process.Start();

        // Setup logging
        if (logDirectory != null)
        {
            Directory.CreateDirectory(logDirectory);
            var logPath = Path.Combine(logDirectory, $"rank_{rank}.log");
            var logWriter = new StreamWriter(logPath);
            _logWriters[rank] = logWriter;

            process.OutputDataReceived += (sender, e) =>
            {
                if (e.Data != null)
                {
                    Console.WriteLine($"[Rank {rank}] {e.Data}");
                    logWriter.WriteLine(e.Data);
                }
            };

            process.ErrorDataReceived += (sender, e) =>
            {
                if (e.Data != null)
                {
                    Console.Error.WriteLine($"[Rank {rank} ERROR] {e.Data}");
                    logWriter.WriteLine($"ERROR: {e.Data}");
                }
            };

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
        }

        _processes[rank] = process;
    }

    /// <summary>
    /// Wait for all processes to complete.
    /// </summary>
    public void WaitForAll()
    {
        for (int i = 0; i < _numProcesses; i++)
        {
            _processes[i].WaitForExit();
            CloseLogWriter(i);
        }
    }

    /// <summary>
    /// Terminate all processes.
    /// </summary>
    public void TerminateAll()
    {
        for (int i = 0; i < _numProcesses; i++)
        {
            if (!_processes[i].HasExited)
            {
                _processes[i].Kill();
            }
            CloseLogWriter(i);
        }
    }

    /// <summary>
    /// Get the exit code for a specific rank.
    /// </summary>
    public int GetExitCode(int rank)
    {
        return _processes[rank].ExitCode;
    }

    /// <summary>
    /// Check if any process has failed.
    /// </summary>
    public bool HasFailedProcess()
    {
        return _processes.Any(p => p.HasExited && p.ExitCode != 0);
    }

    /// <summary>
    /// Get the first failed process information.
    /// </summary>
    public (int rank, int exitCode) GetFirstFailedProcess()
    {
        for (int i = 0; i < _numProcesses; i++)
        {
            if (_processes[i].HasExited && _processes[i].ExitCode != 0)
            {
                return (i, _processes[i].ExitCode);
            }
        }
        return (-1, 0);
    }

    private void CloseLogWriter(int rank)
    {
        _logWriters[rank]?.Close();
        _logWriters[rank] = null;
    }

    public void Dispose()
    {
        TerminateAll();
    }
}
```

### 2. DistributedLauncher Class (High-Level API)
```csharp
/// <summary>
/// High-level API for launching distributed training.
/// </summary>
public static class DistributedLauncher
{
    /// <summary>
    /// Launch distributed training on localhost.
    /// </summary>
    public static void LaunchLocal(
        string executablePath,
        int numProcesses,
        string[] args,
        string logDirectory = "./logs")
    {
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
                    $"Process rank {rank} failed with exit code {exitCode}");
            }
        }
        finally
        {
            launcher.Dispose();
        }
    }

    /// <summary>
    /// Launch distributed training on multiple nodes (multi-node).
    /// </summary>
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
```

### 3. DistributedTrainingException Class
```csharp
public class DistributedTrainingException : Exception
{
    public int Rank { get; }
    public int ExitCode { get; }

    public DistributedTrainingException(string message, int rank = -1, int exitCode = -1)
        : base(message)
    {
        Rank = rank;
        ExitCode = exitCode;
    }
}
```

## Implementation Details

### Environment Variables

Each process is launched with these environment variables:

```
RANK = <rank>                          # 0, 1, 2, ..., world_size-1
WORLD_SIZE = <num_processes>           # Total number of processes
MASTER_ADDR = <master_address>         # IP address of rank 0
MASTER_PORT = <master_port>            # Port for initialization
CUDA_VISIBLE_DEVICES = <rank>           # GPU for this process
```

### CUDA Device Assignment

Each process gets exclusive access to one GPU:

```csharp
// Rank 0 sees GPU 0, rank 1 sees GPU 1, etc.
CUDA_VISIBLE_DEVICES = rank
```

This ensures that:
- Each process sees its GPU as device 0
- Processes don't conflict over GPU access
- Works with any number of GPUs

### Log Management

**Output Logging**:
- Each process's stdout is logged to `logs/rank_<rank>.log`
- Console shows real-time output with rank prefix: `[Rank 0] ...`

**Error Logging**:
- Errors are logged with `[Rank <rank> ERROR]` prefix
- Helps debug which process failed

### Process Management

**Startup**:
- All processes start simultaneously
- Wait for all processes to initialize
- Detect failures early

**Shutdown**:
- Wait for all processes to complete normally
- On failure, terminate all processes
- Clean up log files

**Status Monitoring**:
- Check if any process has failed
- Get exit codes for all processes
- Detect hanging processes

### Multi-Node Support

**Current Implementation**: Localhost only

**Future Multi-Node**: Options for launching on multiple nodes:

1. **SSH-based**: Launch processes on remote nodes via SSH
2. **SLURM Integration**: Use SLURM job scheduler for HPC clusters
3. **MPI-based**: Use MPI launcher (mpirun) to spawn processes

For this spec, we implement localhost launching first. Multi-node can be added later.

## Usage Examples

### Localhost Training

```csharp
// Launch training script on 4 GPUs
DistributedLauncher.LaunchLocal(
    executablePath: "dotnet",
    numProcesses: 4,
    args: new[] { "run", "--project", "TrainModel.csproj" },
    logDirectory: "./logs");
```

### With Custom Launcher

```csharp
var launcher = new ProcessLauncher(
    executablePath: "dotnet",
    numProcesses: 8,
    masterAddr: "127.0.0.1",
    masterPort: 29500);

try
{
    launcher.Launch(new[] { "run", "TrainModel.csproj" }, "./logs");

    // Monitor while running
    while (!launcher.HasFailedProcess())
    {
        Thread.Sleep(1000);
        // Could check progress, etc.
    }

    launcher.WaitForAll();
}
finally
{
    launcher.Dispose();
}
```

### Multi-Node (Future)

```csharp
DistributedLauncher.LaunchMultiNode(
    executablePath: "dotnet",
    numNodes: 2,
    processesPerNode: 4,
    args: new[] { "run", "TrainModel.csproj" },
    masterAddr: "node0",
    masterPort: 29500,
    logDirectory: "./logs");
```

## Success Criteria
- [ ] Launcher correctly sets environment variables
- [ ] Each process is assigned to a unique GPU
- [ ] All processes start simultaneously
- [ ] Log files are created correctly
- [ ] Process failures are detected and reported
- [ ] Cleanup works correctly on success and failure
- [ ] Works with various numbers of GPUs

## Dependencies
- No other DDP specs required (standalone utility)
- System.Diagnostics.Process for process management
- .NET runtime (for launching .NET applications)

## Testing
- Unit tests in spec_ddp_tests.md will verify:
  - Correct environment variable setup
  - Process lifecycle (start, wait, terminate)
  - Log file creation
  - Failure detection
  - Multi-GPU assignment
  - Cleanup behavior

## Notes

- This launcher is similar to `torch.distributed.launch` or `torchrun`
- For production use, consider integrating with job schedulers (SLURM, PBS)
- Multi-node support requires additional infrastructure (SSH, job scheduler)
- On Windows, process launching works the same way as on Linux
