namespace MLFramework.Functional.Distributed
{
    public static class DeviceMeshFactory
    {
        /// <summary>
        /// Create a 1D mesh of devices.
        /// </summary>
        public static DeviceMesh Create1D(int deviceCount, DeviceType type = DeviceType.CPU)
        {
            var devices = new Device[deviceCount];
            for (int i = 0; i < deviceCount; i++)
            {
                devices[i] = type == DeviceType.CPU ? Device.CPU(i) : Device.GPU(i);
            }
            return new DeviceMesh(devices);
        }

        /// <summary>
        /// Create a 2D mesh of devices (e.g., for data parallelism and model parallelism).
        /// </summary>
        public static DeviceMesh Create2D(int rows, int cols, DeviceType type = DeviceType.CPU)
        {
            int totalDevices = rows * cols;
            var devices = new Device[totalDevices];
            for (int i = 0; i < totalDevices; i++)
            {
                devices[i] = type == DeviceType.CPU ? Device.CPU(i) : Device.GPU(i);
            }
            return new DeviceMesh(new int[] { rows, cols }, devices);
        }

        /// <summary>
        /// Get default mesh for the system.
        /// </summary>
        public static DeviceMesh Default()
        {
            // In reality, this would query available devices
            // For now, return a single CPU mesh
            return Create1D(1, DeviceType.CPU);
        }
    }
}
