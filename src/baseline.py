# Threshold-Based Baseline

# Purpose:
# Identify underutilised virtual machines using threshold values based on
# the published Azure Advisor shutdown recommendation.

# The baseline will later be compared against the Random Forest model.

from pathlib import Path
import pandas as pd


# Load the cleaned dataset

data_folder = Path(__file__).parent.parent / "data"

vm_data = pd.read_csv(
    data_folder / "cleaned_vm_features.csv"
)

print(f"Loaded {len(vm_data)} virtual machines.")


# Rule 1: P95 CPU utilisation below 3%

# checks every VM to see if its P95 CPU is below 3%
vm_data["rule_1_cpu"] = (
    vm_data["cpu_p95"] < 3
)

print("\nRule 1: P95 CPU below 3%")
print(
    "VMs meeting Rule 1:",
    vm_data["rule_1_cpu"].sum(),
)
print(
    "VMs not meeting Rule 1:",
    (~vm_data["rule_1_cpu"]).sum(),
)


# Rule 2: Average CPU over the last 3 days at or below 2%

vm_data["rule_2_cpu"] = (
    vm_data["cpu_last_3_days_mean"] <= 2
)

print("\nRule 2: Last 3 days average CPU at or below 2%")
print(
    "VMs meeting Rule 2:",
    vm_data["rule_2_cpu"].sum(),
)
print(
    "VMs not meeting Rule 2:",
    (~vm_data["rule_2_cpu"]).sum(),
)


# Check the outbound network values before finalising Rule 3

print("\nOutbound network summary:")
print(
    vm_data["outbound_network_mean"].describe()
)

print("\nOutbound network percentiles:")
print(
    vm_data["outbound_network_mean"].quantile(
        [0.05, 0.25, 0.50, 0.75]
    )
)


# Rule 3: Average outbound network below 2 KB/s

# checks if the VM sends less than 2 KB/s on average
vm_data["rule_3_network"] = (
    vm_data["outbound_network_mean"] < 2
)

print("\nRule 3: Average outbound network below 2 KB/s")
print(
    "VMs meeting Rule 3:",
    vm_data["rule_3_network"].sum(),
)
print(
    "VMs not meeting Rule 3:",
    (~vm_data["rule_3_network"]).sum(),
)


# Final baseline decision

# a VM is marked as underutilised only if it passes all 3 rules
vm_data["baseline_prediction"] = (
    vm_data["rule_1_cpu"]
    & vm_data["rule_2_cpu"]
    & vm_data["rule_3_network"]
)

print("\nFinal baseline results")
print(
    "Underutilised VMs:",
    vm_data["baseline_prediction"].sum(),
)
print(
    "Not underutilised VMs:",
    (~vm_data["baseline_prediction"]).sum(),
)


# Save the baseline results

output_file = (
    data_folder / "baseline_predictions.csv"
)

vm_data.to_csv(
    output_file,
    index=False,
)

print("\nBaseline predictions saved to:")
print(output_file)