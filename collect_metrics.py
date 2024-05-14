import os
import psutil
import csv
from datetime import datetime
import time

# Create a folder if it doesn't exist
data_folder = "data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# File path for the CSV file in the data folder
csv_file = os.path.join(data_folder, "system_metrics_1.csv")

# Set to keep track of processes already written
# written_processes = set()


# Function to collect system metrics
def collect_metrics():
    # Open CSV file for writing
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Process ID",
                "Process Name",
                "Timestamp",
                "CPU Usage (%)",
                "Memory Usage (%)",
                "Disk Usage (%)",
                "Anomaly",
            ]
        )
        end = (datetime.now().timestamp() + 300) * (10**6)  # 5 minutes from now

        timestamp = datetime.now().timestamp() * (10**6)
        # Collect and write metrics to the CSV file
        while timestamp < end:
            timestamp = datetime.now().timestamp() * (10**6)
            for proc in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_percent"]
            ):
                process_id = proc.info["pid"]
                process_name = proc.info["name"]
                cpu_usage = proc.info["cpu_percent"]
                mem_usage = proc.info["memory_percent"]
                disk_usage = estimate_disk_usage(process_id)
                anomaly = check_for_anomaly(cpu_usage, mem_usage, disk_usage)

                writer.writerow(
                    [
                        process_id,
                        process_name,
                        timestamp,
                        cpu_usage,
                        mem_usage,
                        disk_usage,
                        anomaly,
                    ]
                )
                # Add process to set of written processes
                # written_processes.add(process_id)
            # time.sleep(5)


# Function to estimate disk usage for a process
def estimate_disk_usage(process_id):
    try:
        process = psutil.Process(process_id)
        files = process.open_files()
        total_size = sum(os.path.getsize(f.path) for f in files)
        return total_size / (
            1024 * 1024
        )  # Convert bytes to megabytes for disk usage estimation
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return (
            0  # Return 0 if the process doesn't exist or if there are permission issues
        )


# Function to check for anomalies based on thresholds
def check_for_anomaly(cpu_usage, mem_usage, disk_usage):

    cpu_threshold = 40
    mem_threshold = 60
    disk_threshold = 80

    # Check if CPU or memory usage exceeds thresholds
    if (
        cpu_usage > cpu_threshold
        or mem_usage > mem_threshold
        or disk_usage > disk_threshold
    ):
        return 1
    else:
        return 0


# Call the function to collect metrics
collect_metrics()
