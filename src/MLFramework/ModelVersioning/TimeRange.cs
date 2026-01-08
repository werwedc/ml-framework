using System.Text.Json.Serialization;
using System.ComponentModel.DataAnnotations;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Defines a time range for routing rules
    /// </summary>
    public class TimeRange
    {
        /// <summary>
        /// The start time of the time range
        /// </summary>
        [JsonPropertyName("startTime")]
        public TimeSpan StartTime { get; set; }

        /// <summary>
        /// The end time of the time range
        /// </summary>
        [JsonPropertyName("endTime")]
        public TimeSpan EndTime { get; set; }

        /// <summary>
        /// Days of the week when this time range is active
        /// </summary>
        [JsonPropertyName("daysOfWeek")]
        public DayOfWeek[] DaysOfWeek { get; set; }

        /// <summary>
        /// Creates a new TimeRange
        /// </summary>
        public TimeRange()
        {
            DaysOfWeek = Array.Empty<DayOfWeek>();
        }

        /// <summary>
        /// Validates that the time range is valid
        /// </summary>
        /// <returns>True if the time range is valid, otherwise false</returns>
        public bool IsValid()
        {
            return StartTime < EndTime && DaysOfWeek != null && DaysOfWeek.Length > 0;
        }

        /// <summary>
        /// Checks if the given datetime falls within this time range
        /// </summary>
        /// <param name="dateTime">The datetime to check</param>
        /// <returns>True if the datetime is within the range, otherwise false</returns>
        public bool IsInRange(DateTime dateTime)
        {
            var timeOfDay = dateTime.TimeOfDay;

            // Check if time is within range (handles overnight ranges)
            if (StartTime <= EndTime)
            {
                if (timeOfDay < StartTime || timeOfDay > EndTime)
                    return false;
            }
            else
            {
                // Overnight range (e.g., 22:00 - 06:00)
                if (timeOfDay < StartTime && timeOfDay > EndTime)
                    return false;
            }

            // Check if day of week matches
            return DaysOfWeek.Contains(dateTime.DayOfWeek);
        }

        /// <summary>
        /// Creates a string representation of the time range
        /// </summary>
        public override string ToString()
        {
            return $"TimeRange(Start: {StartTime}, End: {EndTime}, Days: {string.Join(", ", DaysOfWeek)})";
        }
    }
}
