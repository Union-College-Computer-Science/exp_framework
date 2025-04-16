"""
Code for making plots based on SNN logs.

Author: Thomas Breimer
April 9th, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import *

def load_logs(file_path):
    """Load CSV log data."""
    return pd.read_csv(file_path)


def plot_neuron_logs(df, xlim, snn_id, layer, neuron_id):
    """Plot logs for a specific neuron from dataframe."""
    
    neuron_df = df[(df['SNN'] == snn_id) &
                   (df['layer'] == layer) &
                   (df['neuron'] == neuron_id)]

    if neuron_df.empty:
        print("No data found for this neuron.")
        return

    logs = {}
    for log_type in ['levellog', 'firelog', 'dutycyclelog']:
        row = neuron_df[neuron_df['log'] == log_type]
        if not row.empty:
            logs[log_type] = row.iloc[0, 4:].astype(float).values
        else:
            logs[log_type] = None

    steps = range(len(logs['levellog']))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    if logs['levellog'] is not None:
        ax1.plot(steps, logs['levellog'], label='Activation Level (levellog)', alpha=0.7)

    if logs['firelog'] is not None:
        spike_steps = [i for i, v in enumerate(logs['firelog']) if v > 0]
        ymin, ymax = ax1.get_ylim()
        spike_ymin = ymin + 0.05 * (ymax - ymin)
        spike_ymax = spike_ymin + 0.1 * (ymax - ymin)
        for spike in spike_steps:
            ax1.vlines(x=spike, ymin=spike_ymin, ymax=spike_ymax, color='red', linewidth=1.5)

    ax1.set_ylabel('Activation Level')
    ax1.set_title(f'SNN {snn_id} | Layer: {layer} | Neuron: {neuron_id}')
    ax1.legend()
    ax1.grid(True)

    if logs['dutycyclelog'] is not None:
        ax2.plot(steps, logs['dutycyclelog'], label='Duty Cycle (dutycyclelog)', color='purple', alpha=0.7)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Duty Cycle')
    ax2.set_xlabel('Timestep')
    ax2.legend()
    ax2.grid(True)

    plt.xlim(0, xlim)
    plt.tight_layout()
    plt.show()

def plot_snn_spiketrains(df, xlim, snn_id):
    """Plot spike trains for all neurons in an SNN with colored lines per neuron."""
    snn_df = df[(df['SNN'] == snn_id) & (df['log'] == 'firelog')]
    snn_df = snn_df.sort_values(by=['layer', 'neuron']).reset_index(drop=True)

    num_neurons = len(snn_df)
    palette = sns.color_palette("husl", num_neurons)

    plt.figure(figsize=(12, num_neurons * 0.5 + 2))

    yticks = []
    yticklabels = []

    for idx, (_, row) in enumerate(snn_df.iterrows()):
        spikes = row.iloc[4:].astype(float).values
        spike_steps = [i for i, v in enumerate(spikes) if v > 0]
        plt.vlines(spike_steps, idx, idx + 1, color=palette[idx], linewidth=1.5)

        yticks.append(idx + 0.5)
        yticklabels.append(f"{row['layer']}-{row['neuron']}")

    plt.yticks(yticks, yticklabels)
    plt.xlabel('Timestep')
    plt.ylabel('Neuron')
    plt.title(f'SNN {snn_id} Spike Trains')
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.xlim(0, xlim)
    plt.show()

def plot_snn_activation_levels(df, xlim, snn_id):
    """Plot activation levels for all neurons in an SNN."""
    snn_df = df[(df['SNN'] == snn_id) & (df['log'] == 'levellog')]

    plt.figure(figsize=(12, 6))

    for _, row in snn_df.iterrows():
        levels = row.iloc[4:].astype(float).values
        label = f"{row['layer']}-{row['neuron']}"
        plt.plot(levels, label=label, alpha=0.7)

    plt.title(f'SNN {snn_id} Activation Levels')
    plt.xlabel('Timestep')
    plt.ylabel('Activation Level')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0, xlim)
    plt.show()


def plot_snn_dutycycles(df, xlim, snn_id):
    """Plot duty cycles for all neurons in an SNN."""
    snn_df = df[(df['SNN'] == snn_id) & (df['log'] == 'dutycyclelog')]

    plt.figure(figsize=(12, 6))

    for _, row in snn_df.iterrows():
        duty = row.iloc[4:].astype(float).values
        label = f"{row['layer']}-{row['neuron']}"
        plt.plot(duty, label=label, alpha=0.7)

    plt.title(f'SNN {snn_id} Duty Cycles')
    plt.xlabel('Timestep')
    plt.ylabel('Duty Cycle')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0, xlim)
    plt.show()