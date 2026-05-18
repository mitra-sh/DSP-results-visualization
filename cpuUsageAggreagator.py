import pandas as pd
import glob
from collections import defaultdict
import numpy as np

# === User-defined parameters ===
cycle_duration_sec = 30   # Duration of each cycle in seconds
x_axis_unit = 'minutes'   # Choose 'minutes' or 'seconds'
cycle_start = 0           # First cycle to include
cycle_end = 50            # Last cycle to include

# Define which VMs to plot
vms_to_plot = [
    "as00750-02", "as00750-03", "as00750-04", "as00750-05", "as00750-06",
    "as00750-07", "as00750-08", "as00750-09", "as00750-10", "as00750-11", "as00750-12", "as00750-13", "as00750-14"
              , "as00750-15", "as00750-16", "as00750-17", "as00750-18", "as00750-19", "as00750-20", "as00750-21"]


# A list of colors for each segment, in order
emit_rate_colors = [
    '#a6cee3',  # Light blue
    '#fdbf6f',  # Light orange
    '#fb9a99',  # Light red
    '#cab2d6',  # Light purple
    '#ffff99',  # Light yellow
    '#a6bddb',  # Light blue-gray
    '#33a02c',  # Light green
]

# Define which line color to use for each approach name
approach_color_map = {
    "heuristic": "blue",
    "staticWeights": "green",
    "globalAdaptationOnly": "red",
    "noLocal_noGlobal": "purple",
    "storm_replication": "black"
}

# === Locate CSV files ===
file_list = glob.glob('cpuUsage_*.csv')
if not file_list:
    print("No CSV files found matching 'cpuUsage_*.csv'")
    exit(1)

# === Data structures ===
vm_data_dict = {}                # { vm_name: [ (approach, df), (approach, df), ... ] }
event_dict = defaultdict(list)   # { time_value: [event_text1, event_text2, ...] }
event_times_set = set()          # For drawing vertical lines


# === Function to aggregate and save data ===
def aggregate_and_save(input_csv, approach_name):
    """
    Aggregates CPU usage by VM and cycle, saving the result to a new CSV file.
    Ensures EmitRate is handled as an integer for VM rows and formatted correctly in the output CSV.

    Args:
        input_csv (str): Path to the input CSV file.
        approach_name (str): Name of the approach (e.g., 'heuristic').
    """
    # Read the input CSV
    df = pd.read_csv(input_csv)

    # Separate VM rows (where 'VM Name' is not NaN) and event rows (where 'VM Name' is NaN and 'Event' is not NaN)
    vm_rows = df[df['VM Name'].notna()].copy()
    event_rows = df[df['VM Name'].isna() & df['Event'].notna()].copy()

    # Convert EmitRate to integer for vm_rows, filling NaN with 0
    vm_rows['EmitRate'] = pd.to_numeric(vm_rows['EmitRate'], errors='coerce').fillna(0).astype(int)

    # Aggregate VM rows by 'VM Name' and 'cycleCount'
    vm_aggregated = vm_rows.groupby(['VM Name', 'cycleCount'], as_index=False).agg({
        'CPU usage': 'mean',                  # Average CPU usage
        'EmitRate': 'first',                  # Take the first EmitRate (now ensured to be int)
        'Event': lambda x: '; '.join(x.dropna().unique())  # Concatenate unique non-empty events
    })

    # Combine aggregated VM data with event rows
    combined_df = pd.concat([vm_aggregated, event_rows], ignore_index=True)

    # Sort by 'cycleCount' and 'VM Name', with NaN 'VM Name' at the end
    combined_df = combined_df.sort_values(by=['cycleCount', 'VM Name'], na_position='last')

    # Format 'EmitRate' to be integer strings for VM rows and empty for event rows
    combined_df['EmitRate'] = combined_df['EmitRate'].apply(lambda x: str(int(x)) if pd.notnull(x) else '')

    # Define output filename
    output_csv = f"aggregated_CPU_usage_{approach_name}.csv"

    # Write to CSV without index
    combined_df.to_csv(output_csv, index=False)

    print(f"Aggregated data saved to {output_csv}")

# === Main Processing Loop ===
for fname in file_list:
    # Extract approach name (e.g., "heuristic" from "cpuUsage_heuristic.csv")
    approach_name = fname.split('cpuUsage_')[1].split('.csv')[0]

    # Create aggregated CSV
    aggregate_and_save(fname, approach_name)

    # Read the original CSV for further processing
    cpu_data = pd.read_csv(fname)

    # Handle EmitRate column
    if 'EmitRate' in cpu_data.columns and cpu_data['EmitRate'].notnull().any():
        cpu_data['EmitRate'] = pd.to_numeric(cpu_data['EmitRate'], errors='coerce').fillna(0).astype(int)
    else:
        cpu_data['EmitRate'] = 0

    # Convert CPU usage to percent
    cpu_data['CPU usage'] = cpu_data['CPU usage'] * 100.0

    # Filter cycles
    cpu_data = cpu_data[
        (cpu_data['cycleCount'] >= cycle_start) &
        (cpu_data['cycleCount'] <= cycle_end)
    ].copy()
    if cpu_data.empty:
        continue

    # Convert cycle to Time
    cpu_data['Time'] = cpu_data['cycleCount'] * cycle_duration_sec
    if x_axis_unit == 'minutes':
        cpu_data['Time'] /= 60.0

    cpu_data.reset_index(drop=True, inplace=True)

    # Collect event info
    events_df = cpu_data[
        cpu_data['Event'].notnull() &
        (cpu_data['Event'].astype(str).str.strip() != '')
    ]
    for idx_e in events_df.index:
        txt = cpu_data.loc[idx_e, 'Event']
        tval = cpu_data.loc[idx_e, 'Time']
        if np.isfinite(tval):
            event_dict[tval].append(txt)
            event_times_set.add(tval)

    # Group by VM
    vm_names = cpu_data['VM Name'].dropna().unique()
    for vm in vm_names:
        df_vm = cpu_data[cpu_data['VM Name'] == vm].copy()

        # Average CPU usage at each time
        df_vm_grouped = df_vm.groupby('Time', as_index=False).agg({
            'CPU usage': 'mean',
            'EmitRate': 'first'
        })
        df_vm_grouped.sort_values('Time', inplace=True, ignore_index=True)

        # Mark segments
        df_vm_grouped['EmitRate_Change'] = df_vm_grouped['EmitRate'].ne(
            df_vm_grouped['EmitRate'].shift(fill_value=np.nan)
        )
        df_vm_grouped['Segment_ID'] = df_vm_grouped['EmitRate_Change'].cumsum()

        if vm not in vm_data_dict:
            vm_data_dict[vm] = []
        vm_data_dict[vm].append((approach_name, df_vm_grouped))

# === Notes ===
# At this point, vm_data_dict contains processed data that could be used for plotting or further analysis.
# You can add visualization code here if desired, using vm_data_dict, event_dict, and event_times_set.





