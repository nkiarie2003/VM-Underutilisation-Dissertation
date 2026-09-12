"""
Baseline Comparison

Compares the threshold baseline against the final reference labels.

The baseline represents the simpler Azure Advisor-style threshold method.
The reference label represents the richer multi-metric decision made earlier.
"""

from pathlib import Path
import pandas as pd


# find the data folder

data_folder = Path(__file__).parent.parent / "data"


# load the final reference-labelled dataset

input_file = data_folder / "reference_labels.csv"

vm_data = pd.read_csv(input_file)

print(f"Loaded {len(vm_data)} virtual machines.")


# check the columns needed for the comparison

required_columns = [
    "vm_id",
    "cpu_p95",
    "cpu_last_3_days_mean",
    "outbound_network_mean",
    "reference_label",
    "cpu_mean",
    "cpu_max",
    "cpu_below_5_pct",
    "memory_mean",
    "disk_mean",
    "possible_bursty",
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


# recreate the threshold baseline

vm_data["baseline_label"] = (
    (vm_data["cpu_p95"] < 3)
    & (vm_data["cpu_last_3_days_mean"] <= 2)
    & (vm_data["outbound_network_mean"] < 2)
)


# count the two sets of underutilised VMs

baseline_underutilised = int(
    vm_data["baseline_label"].sum()
)

reference_underutilised = int(
    vm_data["reference_label"].sum()
)


print("\nBaseline results:")
print(
    "Baseline underutilised VMs:",
    baseline_underutilised,
)

print(
    "Reference underutilised VMs:",
    reference_underutilised,
)


# compare both labels for every VM

vm_data["baseline_agrees"] = (
    vm_data["baseline_label"]
    == vm_data["reference_label"]
)


# count agreements and disagreements

agreement_count = int(
    vm_data["baseline_agrees"].sum()
)

disagreement_count = int(
    (~vm_data["baseline_agrees"]).sum()
)


# calculate percentages

agreement_percentage = (
    vm_data["baseline_agrees"].mean()
    * 100
)

disagreement_percentage = (
    100 - agreement_percentage
)


print("\nAgreement results:")

print(
    "Agreements:",
    agreement_count,
)

print(
    "Disagreements:",
    disagreement_count,
)

print(
    "Agreement percentage:",
    round(agreement_percentage, 1),
    "%",
)

print(
    "Disagreement percentage:",
    round(disagreement_percentage, 1),
    "%",
)


# create the crosstab

comparison_table = pd.crosstab(
    vm_data["baseline_label"],
    vm_data["reference_label"],
    rownames=["Baseline"],
    colnames=["Reference"],
)

print(
    "\nCrosstab "
    "(rows = baseline, columns = reference):"
)

print(comparison_table)


# baseline says underutilised but reference says no

false_positives = vm_data[
    vm_data["baseline_label"]
    & ~vm_data["reference_label"]
].copy()


# baseline says no but reference says underutilised

false_negatives = vm_data[
    ~vm_data["baseline_label"]
    & vm_data["reference_label"]
].copy()


print("\nDisagreement results:")

print(
    "Baseline underutilised, reference not:",
    len(false_positives),
)

print(
    "Baseline not underutilised, reference yes:",
    len(false_negatives),
)


# columns useful for understanding disagreements

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
    "possible_bursty",
    "baseline_label",
    "reference_label",
]


print(
    "\nFirst 20 VMs where the baseline "
    "says underutilised but reference says no:"
)

print(
    false_positives[
        columns_to_check
    ]
    .head(20)
    .round(2)
    .to_string(index=False)
)


print(
    "\nFirst 20 VMs where the baseline "
    "says no but reference says underutilised:"
)

print(
    false_negatives[
        columns_to_check
    ]
    .head(20)
    .round(2)
    .to_string(index=False)
)


# save the full comparison

comparison_output = (
    data_folder
    / "baseline_comparison.csv"
)

vm_data.to_csv(
    comparison_output,
    index=False,
)


# save both disagreement groups

false_positive_output = (
    data_folder
    / "baseline_false_positives.csv"
)

false_positives[
    columns_to_check
].to_csv(
    false_positive_output,
    index=False,
)


false_negative_output = (
    data_folder
    / "baseline_false_negatives.csv"
)

false_negatives[
    columns_to_check
].to_csv(
    false_negative_output,
    index=False,
)


print("\nComparison saved to:")
print(comparison_output)

print("\nFalse positives saved to:")
print(false_positive_output)

print("\nFalse negatives saved to:")
print(false_negative_output)