from pathlib import Path
import pandas as pd

# loads all the fastStorage csv files and makes one row for each VM

data_folder = Path(__file__).parent.parent / "data" / "fastStorage"
csv_files = list(data_folder.glob("*.csv"))

print("Number of CSV files found:", len(csv_files))

vm_summaries = []
skipped_files = []

for file_path in csv_files:
    try:
        # read one VM file
        vm_data = pd.read_csv(
            file_path,
            sep=";",
            skipinitialspace=True,
        )

        # some column names had extra spaces
        vm_data.columns = vm_data.columns.str.strip()

        # timestamp says ms but the values are actually seconds
        timestamp_values = pd.to_numeric(
            vm_data["Timestamp [ms]"],
            errors="coerce",
        )

        vm_data["timestamp"] = pd.to_datetime(
            timestamp_values,
            unit="s",
            errors="coerce",
        )

        # remove rows where the timestamp did not work
        vm_data = vm_data.dropna(subset=["timestamp"])

        if vm_data.empty:
            raise ValueError("No valid rows found.")

        cpu = vm_data["CPU usage [%]"]

        # get CPU data from the last 3 days for rule 2
        last_3_days_start = (
            vm_data["timestamp"].max()
            - pd.Timedelta(days=3)
        )

        cpu_last_3_days = vm_data.loc[
            vm_data["timestamp"] >= last_3_days_start,
            "CPU usage [%]",
        ]

        cpu_last_3_days_mean = cpu_last_3_days.mean()

        # memory usage as a percentage
        memory_capacity = (
            vm_data["Memory capacity provisioned [KB]"]
            .replace(0, pd.NA)
        )

        memory_percent = (
            vm_data["Memory usage [KB]"]
            / memory_capacity
            * 100
        )

        # add disk reads and writes together
        disk = (
            vm_data["Disk read throughput [KB/s]"]
            + vm_data["Disk write throughput [KB/s]"]
        )

        # total network traffic
        network = (
            vm_data["Network received throughput [KB/s]"]
            + vm_data["Network transmitted throughput [KB/s]"]
        )

        # outbound traffic only for baseline rule 3
        outbound_network = vm_data[
            "Network transmitted throughput [KB/s]"
        ]

        observation_duration = (
            vm_data["timestamp"].max()
            - vm_data["timestamp"].min()
        )

        days_observed = (
            observation_duration.total_seconds() / 86400
        )

        # one summary row for this VM
        vm_summaries.append({
            "vm_id": file_path.stem,
            "samples": len(vm_data),
            "days_observed": days_observed,
            "cpu_cores": vm_data["CPU cores"].iloc[0],
            "cpu_mean": cpu.mean(),
            "cpu_last_3_days_mean": cpu_last_3_days_mean,
            "cpu_p95": cpu.quantile(0.95),
            "cpu_max": cpu.max(),
            "cpu_below_5_pct": (cpu < 5).mean() * 100,
            "memory_mean": memory_percent.mean(),
            "disk_mean": disk.mean(),
            "network_mean": network.mean(),
            "outbound_network_mean": outbound_network.mean(),
        })

    except Exception as error:
        skipped_files.append({
            "file": file_path.name,
            "error": str(error),
        })

        print("Skipped:", file_path.name, "-", error)

vm_features = pd.DataFrame(vm_summaries)

print("\nFirst five VM summaries:")
print(vm_features.head().round(2).to_string())

print("\nShape:")
print(vm_features.shape)

print("\nProcessed:", len(vm_summaries), "of", len(csv_files))

print("\nMissing values:")
print(vm_features.isna().sum())

if skipped_files:
    print("\nSkipped files:")

    for skipped in skipped_files:
        print(skipped["file"], "-", skipped["error"])


# EDA - checking what the dataset looks like

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nSummary Statistics:")
print(vm_features.describe().round(2).to_string())

print("\nMissing Values:")
print(vm_features.isna().sum())

print("\nDuplicate Rows:")
print(vm_features.duplicated().sum())

print("\nData Types:")
print(vm_features.dtypes)

print("\nDataset Shape:")
print(vm_features.shape)

print("\nCPU Mean Range:")
print("Minimum:", round(vm_features["cpu_mean"].min(), 2))
print("Maximum:", round(vm_features["cpu_mean"].max(), 2))

print("\nCPU Last 3 Days Mean Range:")
print(
    "Minimum:",
    round(vm_features["cpu_last_3_days_mean"].min(), 2),
)
print(
    "Maximum:",
    round(vm_features["cpu_last_3_days_mean"].max(), 2),
)

print("\nCPU P95 Range:")
print("Minimum:", round(vm_features["cpu_p95"].min(), 2))
print("Maximum:", round(vm_features["cpu_p95"].max(), 2))

print("\nCPU Max Range:")
print("Minimum:", round(vm_features["cpu_max"].min(), 2))
print("Maximum:", round(vm_features["cpu_max"].max(), 2))

print("\nOutbound Network Mean Range:")
print(
    "Minimum:",
    round(vm_features["outbound_network_mean"].min(), 2),
)
print(
    "Maximum:",
    round(vm_features["outbound_network_mean"].max(), 2),
)

print("\nObservation Period:")
print(vm_features["days_observed"].describe().round(2))


# checks for suspicious values

print("\n" + "=" * 60)
print("DATA QUALITY CHECKS")
print("=" * 60)

print(
    "VMs with cpu_max > 100:",
    (vm_features["cpu_max"] > 100).sum(),
)

print(
    "VMs with cpu_p95 > 100:",
    (vm_features["cpu_p95"] > 100).sum(),
)

print(
    "VMs with memory_mean > 100:",
    (vm_features["memory_mean"] > 100).sum(),
)

print(
    "VMs with cpu_cores == 0:",
    (vm_features["cpu_cores"] == 0).sum(),
)

print(
    "VMs with days_observed < 29:",
    (vm_features["days_observed"] < 29).sum(),
)

print(
    "VMs with samples < 100:",
    (vm_features["samples"] < 100).sum(),
)


# showing the actual rows so they can be inspected

print("\n" + "=" * 60)
print("DATA QUALITY INVESTIGATION")
print("=" * 60)

print("\nVMs with fewer than 100 samples:")
print(vm_features[vm_features["samples"] < 100])

print("\nVMs with memory_mean > 100:")
print(vm_features[vm_features["memory_mean"] > 100])

print("\nFirst 10 VMs with cpu_cores == 0:")
print(
    vm_features[
        vm_features["cpu_cores"] == 0
    ].head(10)
)

print("\nVMs with days_observed < 29:")
print(
    vm_features[
        vm_features["days_observed"] < 29
    ]
)

print("\nFirst 10 VMs with cpu_max > 100:")
print(
    vm_features[
        vm_features["cpu_max"] > 100
    ].head(10)
)


# final cleaning rules

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

cleaned_vm_features = vm_features.copy()
original_count = len(cleaned_vm_features)

low_sample_count = (
    cleaned_vm_features["samples"] < 100
).sum()

# remove VMs with hardly any data
cleaned_vm_features = cleaned_vm_features[
    cleaned_vm_features["samples"] >= 100
].copy()

high_memory_count = (
    cleaned_vm_features["memory_mean"] > 100
).sum()

# remove impossible memory averages
cleaned_vm_features = cleaned_vm_features[
    cleaned_vm_features["memory_mean"] <= 100
].copy()

short_observation_count = (
    cleaned_vm_features["days_observed"] < 7
).sum()

# less than a week is not enough to judge workload behaviour
cleaned_vm_features = cleaned_vm_features[
    cleaned_vm_features["days_observed"] >= 7
].copy()

cleaned_vm_features = (
    cleaned_vm_features
    .reset_index(drop=True)
)

removed_count = (
    original_count
    - len(cleaned_vm_features)
)

print("\nCleaning summary:")
print("Original VMs:", original_count)
print("Low-sample VMs identified:", low_sample_count)
print(
    "Remaining high-memory VMs identified:",
    high_memory_count,
)
print(
    "VMs removed (<7 days observed):",
    short_observation_count,
)
print("Total VMs removed:", removed_count)
print("Remaining VMs:", len(cleaned_vm_features))

print("\nVerification:")
print(
    "VMs with samples < 100:",
    (cleaned_vm_features["samples"] < 100).sum(),
)
print(
    "VMs with memory_mean > 100:",
    (cleaned_vm_features["memory_mean"] > 100).sum(),
)
print(
    "VMs with days_observed < 7:",
    (cleaned_vm_features["days_observed"] < 7).sum(),
)
print(
    "Missing values:",
    cleaned_vm_features.isna().sum().sum(),
)
print(
    "Duplicate rows:",
    cleaned_vm_features.duplicated().sum(),
)

# save the cleaned dataset with the features needed for the baseline
output_file = (
    Path(__file__).parent.parent
    / "data"
    / "cleaned_vm_features.csv"
)

cleaned_vm_features.to_csv(
    output_file,
    index=False,
)

print("\nCleaned dataset saved to:")
print(output_file)


# Save baseline predictions


output_file = (
    data_folder / "baseline_predictions.csv"
)

vm_data.to_csv(
    output_file,
    index=False,
)

print("\nBaseline predictions saved to:")
print(output_file)