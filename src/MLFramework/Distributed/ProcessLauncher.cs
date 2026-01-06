using System.Diagnostics;

namespace MLFramework.Distributed;

/// <summary>
/// Manages the launching and lifecycle of multiple processes for distributed training.
/// </summary>
public class ProcessLauncher : IDisposable
{
    private readonly string _executablePath;
    private readonly int _numProcesses;
    private readonly string _masterAddr;
    private readonly int _masterPort;
    private readonly Process[] _processes;
    private readonly StreamWriter[] _logWriters;

    /// <summary>
    /// Initializes a new instance of the <see cref="ProcessLauncher"/> class.
    /// </summary>
    /// <param name="executablePath">The path to the executable to launch.</param>
    /// <param name="numProcesses">The number of processes to launch.</param>
    /// <param name="masterAddr">The master address for process initialization.</param>
    /// <param name="masterPort">The master port for process initialization.</param>
    public ProcessLauncher(
        string executablePath,
        int numProcesses,
        string masterAddr = "127.0.0.1",
        int masterPort = 29500)
    {
        _executablePath = executablePath ?? throw new ArgumentNullException(nameof(executablePath));
        if (numProcesses <= 0)
        {
            throw new ArgumentException("Number of processes must be greater than 0", nameof(numProcesses));
        }

        _numProcesses = numProcesses;
        _masterAddr = masterAddr ?? "127.0.0.1";
        _masterPort = masterPort;
        _processes = new Process[numProcesses];
        _logWriters = new StreamWriter[numProcesses];
    }

    /// <summary>
    /// Launches all processes with the specified arguments.
    /// </summary>
    /// <param name="args">The command-line arguments to pass to each process.</param>
    /// <param name="logDirectory">Optional directory to store process log files.</param>
    public void Launch(string[] args, string? logDirectory = null)
    {
        for (int rank = 0; rank < _numProcesses; rank++)
        {
            LaunchProcess(rank, args, logDirectory);
        }
    }

    /// <summary>
    /// Launches a specific process by rank.
    /// </summary>
    /// <param name="rank">The rank of the process to launch.</param>
    /// <param name="args">The command-line arguments to pass to the process.</param>
    /// <param name="logDirectory">Optional directory to store process log files.</param>
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
        var allArgs = new List<string>(args ?? Array.Empty<string>());
        allArgs.Add($"--distributed-rank={rank}");
        allArgs.Add($"--distributed-world-size={_numProcesses}");
        startInfo.Arguments = string.Join(" ", allArgs);

        // Create process
        var process = new Process { StartInfo = startInfo };
        process.Start();

        // Setup logging
        if (!string.IsNullOrEmpty(logDirectory))
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
    /// Waits for all processes to complete.
    /// </summary>
    public void WaitForAll()
    {
        for (int i = 0; i < _numProcesses; i++)
        {
            _processes[i]?.WaitForExit();
            CloseLogWriter(i);
        }
    }

    /// <summary>
    /// Terminates all processes.
    /// </summary>
    public void TerminateAll()
    {
        for (int i = 0; i < _numProcesses; i++)
        {
            if (_processes[i] != null && !_processes[i].HasExited)
            {
                _processes[i].Kill();
            }
            CloseLogWriter(i);
        }
    }

    /// <summary>
    /// Gets the exit code for a specific rank.
    /// </summary>
    /// <param name="rank">The rank of the process.</param>
    /// <returns>The exit code of the process.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the process has not exited.</exception>
    public int GetExitCode(int rank)
    {
        if (rank < 0 || rank >= _numProcesses)
        {
            throw new ArgumentOutOfRangeException(nameof(rank), $"Rank must be between 0 and {_numProcesses - 1}");
        }

        if (!_processes[rank].HasExited)
        {
            throw new InvalidOperationException($"Process rank {rank} has not exited yet");
        }

        return _processes[rank].ExitCode;
    }

    /// <summary>
    /// Checks if any process has failed.
    /// </summary>
    /// <returns><c>true</c> if any process has exited with a non-zero exit code; otherwise, <c>false</c>.</returns>
    public bool HasFailedProcess()
    {
        return _processes.Any(p => p != null && p.HasExited && p.ExitCode != 0);
    }

    /// <summary>
    /// Gets information about the first failed process.
    /// </summary>
    /// <returns>A tuple containing the rank and exit code of the first failed process, or (-1, 0) if no process has failed.</returns>
    public (int rank, int exitCode) GetFirstFailedProcess()
    {
        for (int i = 0; i < _numProcesses; i++)
        {
            if (_processes[i] != null && _processes[i].HasExited && _processes[i].ExitCode != 0)
            {
                return (i, _processes[i].ExitCode);
            }
        }
        return (-1, 0);
    }

    /// <summary>
    /// Closes the log writer for a specific rank.
    /// </summary>
    /// <param name="rank">The rank of the process.</param>
    private void CloseLogWriter(int rank)
    {
        _logWriters[rank]?.Close();
        _logWriters[rank] = null!;
    }

    /// <summary>
    /// Disposes resources used by the launcher.
    /// </summary>
    public void Dispose()
    {
        TerminateAll();
        GC.SuppressFinalize(this);
    }
}
