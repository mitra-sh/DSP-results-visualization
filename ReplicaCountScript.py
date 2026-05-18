import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import os

# --------------------------------------------------------
#           USER CONFIGURABLE SETTINGS
# --------------------------------------------------------
time_per_cycle_seconds = 30
start_cycle = 0
end_cycle = 40
initial_replicas = 0  # Initial number of replicas

# NEW: Configurable text positioning settings
EMIT_RATE_TEXT_HEIGHT_RATIO = 0.95  # Position text at 95% of plot height (moved up from 85%)
EMIT_RATE_TEXT_SIZE = 9

# FIXED: Increased vertical offset settings for better line separation
VERTICAL_OFFSET_FACTOR = 0.05  # Increased from 0.01
MAX_OFFSET_MULTIPLIER = 0.05   # Increased significantly from 0.01

vm_list = ['as00750-03', 'as00750-04', 'as00750-05','as00750-06','as00750-07', 'as00750-08', 'as00750-09', 
           'as00750-10','as00750-11','as00750-12','as00750-13','as00750-14','as00750-15','as00750-16',
           'as00750-17','as00750-18','as00750-19','as00750-20','as00750-21']

emit_rate_colors = [
    '#a6cee3',  # Light blue
    '#fdbf6f',  # Light orange
    '#fb9a99',  # Light red
    '#cab2d6',  # Light purple
    '#ffff99',  # Light yellow
    '#a6bddb',  # Light blue-gray
    '#33a02c',  # Light green
]

approach_color_map = {
    "heuristic": "blue",
    "globalAdaptationOnly": "red",
    "noLocal_noGlobal": "purple",
    "storm_replication": "green"
}

# --------------------------------------------------------
#   HELPER FUNCTIONS
# --------------------------------------------------------
def get_consistent_text_position(y_max, position_ratio=EMIT_RATE_TEXT_HEIGHT_RATIO):
    """
    Get a consistent y-position for emit rate text labels.
    
    Args:
        y_max: Maximum y-value of the plot
        position_ratio: Ratio of y_max where text should be positioned (0.0-1.0)
    
    Returns:
        Y-position for the text
    """
    return y_max * position_ratio

def calculate_vertical_offset(approach_index, total_approaches, y_range):
    """
    Calculate vertical offset for each approach to prevent line overlap.
    
    Args:
        approach_index: Index of the current approach (0-based)
        total_approaches: Total number of approaches
        y_range: Range of y-values in the plot
    
    Returns:
        Vertical offset value
    """
    if total_approaches == 1:
        return 0
    
    # FIXED: Use a more significant offset calculation
    # Create offsets that are more spaced out
    max_offset = y_range * MAX_OFFSET_MULTIPLIER
    
    # Space offsets more evenly and make them more visible
    if total_approaches > 1:
        step = (max_offset) / (total_approaches )
        offset = -max_offset + (approach_index * step)
    else:
        offset = 0
    
    return offset

def load_and_prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    df['In Bandwidth Percentage'] = (1 - df['Remaining In Bandwidth Percentage']) * 100
    df['Out Bandwidth Percentage'] = (1 - df['Remaining Out Bandwidth Percentage']) * 100
    df['Time (Minutes)'] = df['cycle'] * (time_per_cycle_seconds / 60.0)
    df['Time (Minutes)'] = df['Time (Minutes)'].astype(float)
    df = df[(df['cycle'] >= start_cycle) & (df['cycle'] <= end_cycle)].copy()
    events_df = df[df['Event'].notnull() & (df['Event'].astype(str).str.strip() != '')]
    events_grouped = events_df.groupby('Time (Minutes)')['Event'].apply(list).reset_index()
    events_dict = dict(zip(events_grouped['Time (Minutes)'], events_grouped['Event']))
    return df, events_dict


def calculate_horizontal_offset(approach_index, total_approaches):
    """
    Calculate horizontal offset for each approach to prevent overlap on sloped lines.
    """
    if total_approaches == 1:
        return 0
    
    # Small horizontal shifts (in minutes)
    max_horizontal_offset = 0.1  # 30 seconds offset
    
    if total_approaches > 1:
        step = (2 * max_horizontal_offset) / (total_approaches - 1)
        offset = -max_horizontal_offset + (approach_index * step)
    else:
        offset = 0
    
    return offset

def get_segment_info(df_vm):
    if df_vm.empty:
        return []
    df_vm = df_vm.copy().reset_index(drop=True)
    df_vm['EmitRate_Change'] = df_vm['EmitRate'].ne(df_vm['EmitRate'].shift())
    df_vm['Segment_ID'] = df_vm['EmitRate_Change'].cumsum()
    segments = df_vm['Segment_ID'].unique()
    seg_list = []
    for idx_s, segment_id in enumerate(segments):
        seg_data = df_vm[df_vm['Segment_ID'] == segment_id]
        start_time = seg_data['Time (Minutes)'].iloc[0]
        if idx_s < len(segments) - 1:
            end_time = df_vm[df_vm['Segment_ID'] == segments[idx_s + 1]]['Time (Minutes)'].iloc[0]
        else:
            end_time = df_vm['Time (Minutes)'].iloc[-1]
        emit_rate_value = seg_data['EmitRate'].iloc[0]
        seg_list.append({
            'Segment_ID': segment_id,
            'Start_Time': start_time,
            'End_Time': end_time,
            'EmitRate': emit_rate_value
        })
    return seg_list

# --------------------------------------------------------
#   FIND ALL FILES TO COMPARE
# --------------------------------------------------------
file_list = glob.glob('remainingBW_*.csv')
if not file_list:
    raise FileNotFoundError("No CSV files found matching 'remainingBW_*.csv'")

file_list = sorted(file_list)

approaches = []
for file in file_list:
    approach_name = os.path.basename(file).split('remainingBW_')[1].split('.csv')[0]
    df, events_dict = load_and_prepare_data(file)
    if not df.empty:
        df = df[df['VM Name'].isin(vm_list)]
        aggregated = (
            df
            .groupby('Time (Minutes)', as_index=False)
            .agg({
                'In Bandwidth Percentage': ['min', 'max', 'mean'],
                'Out Bandwidth Percentage': ['min', 'max', 'mean'],
                'EmitRate': 'mean'
            })
        )
        aggregated.columns = [
            'Time (Minutes)',
            'min_in', 'max_in', 'avg_in',
            'min_out', 'max_out', 'avg_out',
            'EmitRate'
        ]
        segs = get_segment_info(aggregated)
    else:
        aggregated = pd.DataFrame()
        segs = []
    approaches.append({
        'name': approach_name,
        'data': df,
        'events': events_dict,
        'color': approach_color_map.get(approach_name, 'gray'),
        'aggregated': aggregated,
        'segs': segs
    })

# --------------------------------------------------------
#   EXTRACT REPLICA COUNTS FOR EACH APPROACH
# --------------------------------------------------------
for approach in approaches:
    df = approach['data']
    time_points = sorted(df['Time (Minutes)'].unique())
    replica_df = pd.DataFrame({'Time (Minutes)': time_points})
    
    # Special handling for storm_replication approach
    if approach['name'] == 'storm_replication':
        # Set constant 4 replicas for storm_replication throughout the run
        replica_df['Replica Count'] = 4
        print(f"Setting storm_replication to constant 4 replicas")
    else:
        # Normal replica counting logic for other approaches
        current_count = initial_replicas
        for time in time_points:
            if time in approach['events']:
                events = approach['events'][time]
                for event in events:
                    if "->" in event:
                        current_count += 1
                    elif "×" in event:
                        current_count -= 1
            current_count = max(current_count, 0)
            replica_df.loc[replica_df['Time (Minutes)'] == time, 'Replica Count'] = current_count
    
    approach['replica_df'] = replica_df

# --------------------------------------------------------
#   CREATE FIGURE FOR REPLICA COUNTS
# --------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 8))  # Slightly taller to accommodate offsets

# Use first approach's segments for shading
first_approach = approaches[0]
segs = first_approach['segs']
bg_colors = [emit_rate_colors[i % len(emit_rate_colors)] for i in range(len(segs))]

# Add shading
for i, seg in enumerate(segs):
    emit_rate_val = seg['EmitRate']
    start_t = seg['Start_Time']
    end_t = seg['End_Time']
    if emit_rate_val != 0:
        ax.axvspan(start_t, end_t, facecolor=bg_colors[i], alpha=0.2)

# Calculate y-range for offset calculations
max_replica = max([app['replica_df']['Replica Count'].max() for app in approaches]) if approaches else 0
min_replica = 0  # Assuming replica count doesn't go below 0
y_range = max(max_replica, 1)  # Ensure minimum range of 1

# Plot replica counts with vertical offsets
total_approaches = len(approaches)

print(f"Plotting {total_approaches} approaches with max replica count: {max_replica}")

for i, approach in enumerate(approaches):
    replica_df = approach['replica_df']
    
    max_replicas = replica_df['Replica Count'].max()
    
    if approach['name'] == 'storm_replication':
        vertical_offset = 0  # Baseline at exactly 4
        horizontal_offset = 0
        print(f"Approach {approach['name']}: NO OFFSET (baseline at 4)")
    elif approach['name'] == 'globalAdaptationOnly':
        vertical_offset = -0.07  # Slightly below storm_replication
        horizontal_offset = 0.15
        print(f"Approach {approach['name']}: SMALL NEGATIVE OFFSET (offset = {vertical_offset})")
    elif approach['name'] == 'noLocal_noGlobal':
        vertical_offset = 0   # Small positive offset from 0
        horizontal_offset = 0
        print(f"Approach {approach['name']}: SMALL OFFSET FROM ZERO (offset = {vertical_offset})")
    elif approach['name'] == 'heuristic':
        vertical_offset = 0.0   # Small positive offset from 0
        horizontal_offset = 0
        print(f"Approach {approach['name']}: SMALL OFFSET FROM ZERO (offset = {vertical_offset})")
    else:
        # noLocal_noGlobal gets normal offset calculation
        vertical_offset = calculate_vertical_offset(i, total_approaches, y_range)
        horizontal_offset = calculate_horizontal_offset(i, total_approaches)
        print(f"Approach {approach['name']}: NORMAL OFFSET (max = {max_replicas}, offset = {vertical_offset})")
    

    # Apply offset to replica counts
    replica_counts_with_offset = replica_df['Replica Count'] + vertical_offset
    time_with_offset = replica_df['Time (Minutes)'] + horizontal_offset

    
    ax.plot(
        time_with_offset,
        replica_counts_with_offset,
        label=approach['name'],  # Clean label without offset info
        color=approach['color'],
        marker='^',
        markersize=4,
        linewidth=2
    )

# IMPROVED: Better positioning for emit rate annotations accounting for offsets
max_offset = abs(calculate_vertical_offset(total_approaches-1, total_approaches, y_range)) if total_approaches > 1 else 0
replica_y_max = (max_replica + max_offset) * 1.2 if max_replica > 0 else 1
text_y_replica = get_consistent_text_position(replica_y_max)

for seg in segs:
    mid_time = 0.5 * (seg['Start_Time'] + seg['End_Time'])
    
    ax.text(
        mid_time, text_y_replica,
        f'Eᵣ={int(seg["EmitRate"])}',
        rotation=0, va='center', ha='center', color='red', 
        fontsize=EMIT_RATE_TEXT_SIZE,
        bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="red", alpha=0.8)
    )

# Configure plot
ax.set_xlabel('Time (Minutes)')
ax.set_ylabel('Total Number of Replicas')
ax.set_title('Total Number of Replicas Over Time')
ax.legend(loc='center left')  # Move legend outside plot area
ax.grid(True, linestyle='--', alpha=0.5)

# IMPROVED: Better y-axis limits accounting for larger offsets
min_offset = calculate_vertical_offset(0, total_approaches, y_range) if total_approaches > 1 else 0
max_offset = calculate_vertical_offset(total_approaches-1, total_approaches, y_range) if total_approaches > 1 else 0

y_min = min_offset - 0.2  # Add some padding
y_max = (max_replica + max_offset) * 1.25 if max_replica > 0 else 1

ax.set_ylim(y_min, y_max)

# Set reasonable y-ticks
if max_replica > 0:
    # Create ticks that show the main replica values
    main_ticks = list(range(0, int(max_replica) + 1))
    ax.set_yticks(main_ticks)
    ax.grid(True, which='minor', linestyle=':', alpha=0.3)
else:
    ax.set_yticks([0, 1])

plt.tight_layout()
out_fig_name = f"Replica_Counts_{'_vs_'.join([app['name'] for app in approaches])}.png"
plt.savefig(out_fig_name, dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

print(f"Plot saved as: {out_fig_name}")
print("Vertical offsets applied to separate overlapping lines.")





