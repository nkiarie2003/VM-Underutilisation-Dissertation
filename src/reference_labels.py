"""
Reference Labels

Creates labels for the cleaned Bitbrains VMs.

True means the VM is labelled underutilised.
False means there is not enough evidence to label it underutilised.

These labels are not verified ground truth. They are transparent,
fixed research labels created because Bitbrains does not provide
authoritative underutilisation labels.

The rule uses more information than the threshold baseline so that
the Random Forest is not trained to copy the baseline exactly.
"""

from pathlib import Path
import pandas as pd


# Load the cleaned dataset

data_folder = Path(__file__).parent.parent / "data"
input_file = data_folder / "cleaned_vm_features.csv"

vm_data = pd.read_csv(input_file)

print(f"Loaded {len(vm_data)} virtual machines.")


# Check the columns needed by the rule

required_columns = [
    "vm_id",
    "cpu_mean",
    "cpu_last_3_days_mean",
    "cpu_p95",
    "cpu_max",
    "cpu_below_5_pct",
    "memory_mean",
    "disk_mean",
    "outbound_network_mean",
]

missing_columns = [
    column
    for column in required_columns
    if column not in vm_data.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# CPU thresholds

IDLE_TIME_SHARE = 95
LOW_CPU_MEAN = 2
LOW_RECENT_CPU = 2
LOW_CPU_P95 = 5


# Dataset-based thresholds

# The median represents the middle VM in this dataset.
# Values below the median are treated as relatively low.
LOW_MEMORY = vm_data["memory_mean"].median()
LOW_DISK = vm_data["disk_mean"].median()
LOW_NETWORK_OUT = vm_data["outbound_network_mean"].median()

# The top quarter of CPU maximum values is treated as a high peak.
BURST_PEAK = vm_data["cpu_max"].quantile(0.75)


print("\nThresholds used:")
print("Idle time share:", IDLE_TIME_SHARE, "%")
print("Low CPU mean:", LOW_CPU_MEAN, "%")
print("Low recent CPU:", LOW_RECENT_CPU, "%")
print("Low CPU P95:", LOW_CPU_P95, "%")
print("Burst CPU peak:", round(BURST_PEAK, 2), "%")
print("Low memory:", round(LOW_MEMORY, 2), "%")
print("Low disk:", round(LOW_DISK, 2), "KB/s")
print(
    "Low outbound network:",
    round(LOW_NETWORK_OUT, 2),
    "KB/s",
)


# Individual CPU checks

vm_data["low_cpu_mean"] = (
    vm_data["cpu_mean"] <= LOW_CPU_MEAN
)

vm_data["low_recent_cpu"] = (
    vm_data["cpu_last_3_days_mean"]
    <= LOW_RECENT_CPU
)

vm_data["low_cpu_p95"] = (
    vm_data["cpu_p95"] < LOW_CPU_P95
)

vm_data["mostly_idle_cpu"] = (
    vm_data["cpu_below_5_pct"]
    >= IDLE_TIME_SHARE
)


# Secondary resource checks

vm_data["low_memory"] = (
    vm_data["memory_mean"] <= LOW_MEMORY
)

vm_data["low_disk"] = (
    vm_data["disk_mean"] <= LOW_DISK
)

vm_data["low_outbound_network"] = (
    vm_data["outbound_network_mean"]
    <= LOW_NETWORK_OUT
)


# Count how many secondary resources are low

vm_data["low_secondary_resource_count"] = (
    vm_data[
        [
            "low_memory",
            "low_disk",
            "low_outbound_network",
        ]
    ]
    .sum(axis=1)
)


# Possible bursty or periodic workload

# Mostly idle CPU combined with a comparatively high CPU peak.
vm_data["possible_bursty"] = (
    vm_data["mostly_idle_cpu"]
    & (
        vm_data["cpu_max"]
        >= BURST_PEAK
    )
)


# Core low-CPU evidence

# All four CPU checks must support low utilisation.
vm_data["core_low_cpu"] = (
    vm_data["low_cpu_mean"]
    & vm_data["low_recent_cpu"]
    & vm_data["low_cpu_p95"]
    & vm_data["mostly_idle_cpu"]
)


# Multi-metric low usage

# CPU must be clearly low, and at least two of the three
# secondary resource measurements must also be low.
vm_data["consistently_low_usage"] = (
    vm_data["core_low_cpu"]
    & (
        vm_data["low_secondary_resource_count"]
        >= 2
    )
)


# Final provisional reference label

# Underutilised only when:
# 1. CPU is consistently low.
# 2. At least two secondary resources are low.
# 3. There is no evidence of a large intermittent CPU peak.
vm_data["reference_label"] = (
    vm_data["consistently_low_usage"]
    & ~vm_data["possible_bursty"]
)


# Results

underutilised_count = int(
    vm_data["reference_label"].sum()
)

not_underutilised_count = int(
    (~vm_data["reference_label"]).sum()
)

possible_bursty_count = int(
    vm_data["possible_bursty"].sum()
)

excluded_by_burst_count = int(
    (
        vm_data["consistently_low_usage"]
        & vm_data["possible_bursty"]
    ).sum()
)


print("\nReference label results:")
print("Underutilised VMs:", underutilised_count)
print(
    "Not underutilised VMs:",
    not_underutilised_count,
)
print(
    "Underutilised percentage:",
    round(
        vm_data["reference_label"].mean()
        * 100,
        1,
    ),
    "%",
)

print("\nIntermediate results:")
print(
    "Core low CPU:",
    int(vm_data["core_low_cpu"].sum()),
)
print(
    "Consistently low usage:",
    int(
        vm_data[
            "consistently_low_usage"
        ].sum()
    ),
)
print(
    "Possible bursty:",
    possible_bursty_count,
)
print(
    "Excluded by burst rule:",
    excluded_by_burst_count,
)


# Columns needed for manual checking

columns_to_check = [
    "vm_id",
    "cpu_mean",
    "cpu_last_3_days_mean",
    "cpu_p95",
    "cpu_max",
    "cpu_below_5_pct",
    "memory_mean",
    "disk_mean",
    "outbound_network_mean",
    "low_secondary_resource_count",
    "possible_bursty",
    "consistently_low_usage",
    "reference_label",
]


# Random sample from both final classes

samples = []

for label_value in [True, False]:

    class_rows = vm_data[
        vm_data["reference_label"]
        == label_value
    ]

    if not class_rows.empty:
        samples.append(
            class_rows.sample(
                n=min(15, len(class_rows)),
                random_state=42,
            )
        )

random_sample = pd.concat(
    samples,
    ignore_index=True,
)


# Sample VMs close to the main CPU thresholds

borderline_sample = vm_data[
    vm_data["cpu_mean"].between(
        LOW_CPU_MEAN - 0.5,
        LOW_CPU_MEAN + 0.5,
    )
    | vm_data[
        "cpu_last_3_days_mean"
    ].between(
        LOW_RECENT_CPU - 0.5,
        LOW_RECENT_CPU + 0.5,
    )
    | vm_data["cpu_p95"].between(
        LOW_CPU_P95 - 1,
        LOW_CPU_P95 + 1,
    )
    | vm_data["cpu_max"].between(
        BURST_PEAK - 5,
        BURST_PEAK + 5,
    )
].copy()

if len(borderline_sample) > 30:
    borderline_sample = (
        borderline_sample.sample(
            n=30,
            random_state=42,
        )
    )


# Combine the random and borderline samples

verification_sample = pd.concat(
    [
        random_sample,
        borderline_sample,
    ],
    ignore_index=True,
)

verification_sample = (
    verification_sample
    .drop_duplicates(subset=["vm_id"])
)

verification_sample = (
    verification_sample[
        columns_to_check
    ]
    .copy()
)

verification_sample["manual_label"] = ""
verification_sample["review_notes"] = ""


print("\nVerification sample:")
print(
    verification_sample[
        columns_to_check
    ]
    .round(2)
    .to_string(index=False)
)

# Save the final reference labels

labels_output = (
    data_folder
    / "reference_labels.csv"
)

vm_data.to_csv(
    labels_output,
    index=False,
)

# Save the smaller manual-checking file

verification_output = (
    data_folder
    / "verification_sample.csv"
)

verification_sample.to_csv(
    verification_output,
    index=False,
)

print("\nFinal reference labels saved to:")
print(labels_output)

print("\nVerification sample saved to:")
print(verification_output)