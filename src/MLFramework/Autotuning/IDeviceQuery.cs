namespace MLFramework.Autotuning;

/// <summary>
/// Interface for querying device information
/// </summary>
public interface IDeviceQuery
{
    DeviceInfo GetCurrentDeviceInfo();
}
