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
end_cycle = 60

vm_list = ['as00750-03', 'as00750-04', 'as00750-05','as00750-06','as00750-07', 'as00750-08', 'as00750-09', 
           'as00750-10','as00750-11','as00750-12','as00750-13','as00750-14','as00750-15','as00750-16',
           'as00750-17','as00750-18','as00750-19','as00750-20','as00750-21']
#vm_list = ['as00750-03', 'as00750-18', 'as00750-17',
#           'as00750-19','as00750-21', 'as00750-20', 'as00750-11']
#vm_list = ['as00750-03', 'as00750-18']
desired_order = [
    "noLocal_noGlobal",
    "storm_replication",
    "globalAdaptationOnly",
    "heuristic",
]
# Bandwidth usage threshold for visualization
BW_THRESHOLD = 80  # 80% threshold line

# Colors for emit rate shading (used only by the first approach in individual VM plots)
emit_rate_colors = [
    '#a6cee3',  # Light blue
    '#fdbf6f',  # Light orange
    '#fb9a99',  # Light red
    '#cab2d6',  # Light purple
    '#ffff99',  # Light yellow
    '#a6bddb',  # Light blue-gray
    '#33a02c',  # Light green
]

# Color mapping for approaches, used in all diagrams
approach_color_map = {

    "noLocal_noGlobal": "purple",
    "storm_replication": "green",
    "globalAdaptationOnly": "red",
        "heuristic": "blue"
}

# --------------------------------------------------------
#   HELPER FUNCTIONS
# --------------------------------------------------------
def load_and_prepare_data(csv_file):
    df = pd.read_csv(csv_file)

    # Convert "Remaining" bandwidth to usage percentage
    df['In Bandwidth Percentage'] = (1 - df['Remaining In Bandwidth Percentage']) * 100
    df['Out Bandwidth Percentage'] = (1 - df['Remaining Out Bandwidth Percentage']) * 100

    # Convert cycle to minutes
    df['Time (Minutes)'] = df['cycle'] * (time_per_cycle_seconds / 60.0)
    df['Time (Minutes)'] = df['Time (Minutes)'].astype(float)

    # Filter by cycle range
    df = df[(df['cycle'] >= start_cycle) & (df['cycle'] <= end_cycle)].copy()

    # Group events by time
    events_df = df[df['Event'].notnull() & (df['Event'].astype(str).str.strip() != '')]
    events_grouped = events_df.groupby('Time (Minutes)')['Event'].apply(list).reset_index()
    events_dict = dict(zip(events_grouped['Time (Minutes)'], events_grouped['Event']))

    return df, events_dict

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

# Sort files for consistent ordering
file_list = sorted(file_list)

# Load data for all approaches
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
    name2app = {a["name"]: a for a in approaches}
    ordered = [name2app[n] for n in desired_order if n in name2app]
    extras  = [a for a in approaches if a["name"] not in set(desired_order)]
    approaches = ordered + extras

# --------------------------------------------------------
#   CREATE FIGURE WITH SUBPLOTS FOR INDIVIDUAL VMs
# --------------------------------------------------------
all_vms = set()
for approach in approaches:
    all_vms.update(approach['data']['VM Name'].dropna().unique())

vms_to_plot = [vm for vm in vm_list if vm in all_vms]
num_vms_to_plot = len(vms_to_plot)
if num_vms_to_plot == 0:
    print("No matching VMs found in the specified files.")
    raise SystemExit

fig, axes = plt.subplots(
    nrows=num_vms_to_plot,
    ncols=2,
    figsize=(20, 5 * num_vms_to_plot)
)
fig.suptitle(
    f"Bandwidth Usage Comparison\n"
    f"Approaches: {', '.join([app['name'] for app in approaches])}\n"
    f"Cycles {start_cycle}-{end_cycle}",
    fontsize=16
)
if num_vms_to_plot == 1:
    axes = np.array([axes])  # Make it 2D if single VM

for idx_vm, vm in enumerate(vms_to_plot):
    ax_in = axes[idx_vm, 0]
    ax_out = axes[idx_vm, 1]

    ax_in.set_title(f'{vm} - IN Bandwidth')
    ax_in.set_xlabel('Time (Minutes)')
    ax_in.set_ylabel('Bandwidth Usage (%)')
    ax_in.set_ylim(0, 100)
    ax_in.grid(True, linestyle='--', alpha=0.5)

    ax_out.set_title(f'{vm} - OUT Bandwidth')
    ax_out.set_xlabel('Time (Minutes)')
    ax_out.set_ylabel('Bandwidth Usage (%)')
    ax_out.set_ylim(0, 100)
    ax_out.grid(True, linestyle='--', alpha=0.5)

    for app_idx, approach in enumerate(approaches):
        vm_data = approach['data'][approach['data']['VM Name'] == vm]
        if not vm_data.empty:
            if app_idx == 0:  # Shading only for the first approach
                segs = get_segment_info(vm_data)
                bg_colors = [emit_rate_colors[i % len(emit_rate_colors)] for i in range(len(segs))]
                for i, seg in enumerate(segs):
                    emit_rate_val = seg['EmitRate']
                                    
                    if emit_rate_val != 0:
                        ax_in.axvspan(seg['Start_Time'], seg['End_Time'], facecolor=bg_colors[i], alpha=0.2)
                        ax_out.axvspan(seg['Start_Time'], seg['End_Time'], facecolor=bg_colors[i], alpha=0.2)
                    
                    mid_time = 0.5 * (seg['Start_Time'] + seg['End_Time'])

                    ax_in.text(
                        mid_time, 95,
                        f'Eᵣ={int(emit_rate_val)}',
                        rotation=0, va='top', ha='center', color='red', fontsize=9
                    )
                    ax_out.text(
                        mid_time, 95,
                        f'Eᵣ={int(emit_rate_val)}',
                        rotation=0, va='top', ha='center', color='red', fontsize=9
                    )
           

            # Plot usage lines
            ax_in.plot(
                vm_data['Time (Minutes)'],
                vm_data['In Bandwidth Percentage'],
                label=approach['name'],
                color=approach['color'],
                marker='o'
            )
            ax_out.plot(
                vm_data['Time (Minutes)'],
                vm_data['Out Bandwidth Percentage'],
                label=approach['name'],
                color=approach['color'],
                marker='o'
            )

            # Plot events with approach-specific colors
            for event_time in sorted(approach['events'].keys()):
                if not np.isfinite(event_time):
                    continue
                ax_in.axvline(x=event_time, color=approach['color'], linestyle='--', linewidth=1, alpha=0.5)
                ax_out.axvline(x=event_time, color=approach['color'], linestyle='--', linewidth=1, alpha=0.5)
                multiline_text = "\n".join(approach['events'][event_time])
                ax_in.text(
                    event_time, 1.05,
                    multiline_text,
                    rotation=90,
                    va='bottom',
                    ha='center',
                    color=approach['color'],
                    fontsize=8,
                    transform=ax_in.get_xaxis_transform(),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=approach['color'], alpha=0.3)
                )
                ax_out.text(
                    event_time, 1.05,
                    multiline_text,
                    rotation=90,
                    va='bottom',
                    ha='center',
                    color=approach['color'],
                    fontsize=8,
                    transform=ax_out.get_xaxis_transform(),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=approach['color'], alpha=0.3)
                )

    ax_in.legend()
    ax_out.legend()
 

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
out_fig_name = f"BW Usage_{'_vs_'.join([app['name'] for app in approaches])}.png"
#plt.savefig(out_fig_name)
plt.close(fig)

# --------------------------------------------------------
#   CREATE AGGREGATED USAGE PLOT FOR ALL APPROACHES
# --------------------------------------------------------
num_approaches = len(approaches)
if num_approaches == 0:
    print("No approaches to plot.")
    raise SystemExit

# Calculate global max for y-axis scaling
global_max_in = max([app['aggregated']['max_in'].max() for app in approaches if not app['aggregated'].empty], default=0)
global_max_out = max([app['aggregated']['max_out'].max() for app in approaches if not app['aggregated'].empty], default=0)

# Create figure and subplots
fig, axes = plt.subplots(nrows=num_approaches, ncols=2, figsize=(20, 5 * num_approaches), sharex=True)
if num_approaches == 1:
    axes = [axes]  # Ensure axes is iterable

# Plot data for each approach
for approach_idx, approach in enumerate(approaches):
    ax_in = axes[approach_idx][0]
    ax_out = axes[approach_idx][1]

    # Set titles and labels
    ax_in.set_title(f"IN Bandwidth Utilization ({approach['name']})")

    ax_out.set_title(f"OUT Bandwidth Utilization ({approach['name']})")
    ax_in.set_ylabel('In Bandwidth Usage (%)')
    ax_out.set_ylabel('Out Bandwidth Usage (%)')
    if approach_idx == num_approaches - 1:
        ax_in.set_xlabel('Time (Minutes)')
        ax_out.set_xlabel('Time (Minutes)')
    ax_in.grid(True, linestyle='--', alpha=0.5)
    ax_out.grid(True, linestyle='--', alpha=0.5)

    aggregated = approach['aggregated']
    segs = approach['segs']
    color = approach['color']

    if not aggregated.empty:
        # Plot shading
        bg_colors = [emit_rate_colors[i % len(emit_rate_colors)] for i in range(len(segs))]
        for i, seg in enumerate(segs):
            emit_rate_val = seg['EmitRate']
            start_t = seg['Start_Time']
            end_t = seg['End_Time']
            if emit_rate_val != 0:
                ax_in.axvspan(start_t, end_t, facecolor=bg_colors[i], alpha=0.2)
                ax_out.axvspan(start_t, end_t, facecolor=bg_colors[i], alpha=0.2)

            
            mid_time = 0.5 * (start_t + end_t)
            if emit_rate_val == 0:
                mid_time = start_t + 2
            ax_in.text(
                mid_time, 0.95 * max(100, global_max_in * 1.0),
                f'Eᵣ={int(emit_rate_val)}',
                #f'Eᵣ={int(emit_rate_val)*2}',
                rotation=0, va='top', ha='center', color='red', fontsize=9
            )
            ax_out.text(
                mid_time, 0.95 * max(100, global_max_out * 1.0),
                f'Eᵣ={int(emit_rate_val)}',
                #f'Eᵣ={int(emit_rate_val)*2}',
                rotation=0, va='top', ha='center', color='red', fontsize=9
            )

        # Plot min-max range for IN
        ax_in.fill_between(
            aggregated['Time (Minutes)'],
            aggregated['min_in'],
            aggregated['max_in'],
            color=color,
            alpha=0.15,
            label=f'{approach["name"]}\nUsage Range'
        )
     #   ax_in.plot(
    #        aggregated['Time (Minutes)'],
     #       aggregated['avg_in'],
      #      color=color,
#            label=f'{approach["name"]}: Average',
    #        linestyle='-',
  #          marker='o'
 #       )

        # Plot min-max range for OUT
        ax_out.fill_between(
            aggregated['Time (Minutes)'],
            aggregated['min_out'],
            aggregated['max_out'],
            color=color,
            alpha=0.15,
            label=f'{approach["name"]}\nUsage Range'
        )
       # ax_out.plot(
       #     aggregated['Time (Minutes)'],
       #     aggregated['avg_out'],
       #     color=color,
       #     label=f'{approach["name"]}: Average',
       #     linestyle='-',
       #     marker='o'
       # )

        # ADD HORIZONTAL THRESHOLD LINES AT 80% FOR BOTH IN AND OUT BANDWIDTH
        ax_in.axhline(y=BW_THRESHOLD, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'{BW_THRESHOLD}% Threshold')
        ax_out.axhline(y=BW_THRESHOLD, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'{BW_THRESHOLD}% Threshold')

        # Set y-axis limits
        ax_in.set_ylim(0, max(100, global_max_in * 1.0))
        ax_out.set_ylim(0, max(100, global_max_out * 1.0))

        ax_in.legend(loc='lower right', bbox_to_anchor=(1.0, 0.1))
        ax_out.legend(loc='lower right', bbox_to_anchor=(1.0, 0.1))
      

# Add a super title
fig.suptitle(
    f"Aggregated Bandwidth Usage Across All Nodes",
    fontsize=16
)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.93])

# Save the figure
out_fig_name = f"Aggregated_BWUsage_{'_vs_'.join([app['name'] for app in approaches])}.png"
plt.savefig(out_fig_name)
plt.close(fig)





