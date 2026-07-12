"""
SPEAR link characterization: Wi-Fi packet loss across test conditions.

Compares packet loss between recordings (one CSV per test condition) and
shows loss over time within each recording, so burst events (signal
blockage, antenna rotation) are visible at the moment they happen.

Outputs (saved to Python/Link Plots/):
  - link_loss_comparison.png : bar chart, loss % per condition
  - link_timeseries_<n>.png  : per-file loss % vs time with burst markers

Loss math lives in spear_analysis.py (compute_loss, find_gaps,
compute_loss_timeseries). This script is just config + plotting.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import spear_analysis as sa

# ===========================================================================
# CONFIG - edit this block per run
# ===========================================================================
# Map: CSV filename (relative to Data Logs/, or absolute path) -> condition
# label. One entry per test condition. Replace the placeholder filenames
# with your actual capture files.
LINK_TESTS = {
    "imu_log_100m.csv":        "LOS 100m+",
    "imu_log_bodythencar.csv": "Behind car ~40m",
    "imu_log_parrallel.csv":   "Parallel 30m",
}

BIN_SECONDS   = 1.0   # time-series bin width
BURST_GAP_MIN = 3     # gaps of >= this many consecutive packets are bursts
# ===========================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir    = os.path.join(script_dir, "Link Plots")
os.makedirs(out_dir, exist_ok=True)

BAR_COLOR   = 'steelblue'   # single hue: bars encode one magnitude, not identities
BURST_COLOR = 'firebrick'


def analyze_one(filename, label):
    """Load one recording, compute loss stats, return everything for plotting."""
    df   = sa.load_log(filename)
    loss = sa.compute_loss(df)
    gaps = sa.find_gaps(df)
    ts   = sa.compute_loss_timeseries(df, bin_seconds=BIN_SECONDS)

    # Convert burst timestamps to seconds from start of recording, matching
    # the time-series axis
    bursts = gaps[gaps['gap_size'] >= BURST_GAP_MIN].copy()
    bursts['t_s'] = (bursts['millis_at_gap'] - df['ms'].iloc[0]) / 1000.0

    return {'df': df, 'loss': loss, 'gaps': gaps, 'ts': ts, 'bursts': bursts,
            'label': label, 'filename': filename}


def plot_timeseries(result, out_path):
    """Loss % vs time for one recording, bursts marked as time events."""
    ts     = result['ts']
    bursts = result['bursts']

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(ts['bin_time_s'], ts['loss_pct'],
            color=BAR_COLOR, linewidth=1.0, label=f'loss % per {BIN_SECONDS:.0f}s bin')

    # Burst markers: vertical lines spanning the panel (data-x, axes-y blended
    # transform, same pattern as the Step 2 gate shading). These mark WHEN a
    # burst happened; the line above shows how bad it was. Not a second y-axis.
    if len(bursts):
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.vlines(bursts['t_s'], 0, 1, transform=trans,
                  color=BURST_COLOR, alpha=0.5, linewidth=1.0,
                  label=f'{len(bursts)} bursts (>= {BURST_GAP_MIN} consecutive lost)')
        # Annotate burst sizes only when they are sparse enough to read.
        # During a sustained blockage dozens of bursts cluster together and
        # the labels would just overlap; the line density carries the story.
        if len(bursts) <= 12:
            for _, b in bursts.iterrows():
                ax.annotate(f"{b['gap_size']:.0f}", (b['t_s'], 0.97),
                            xycoords=trans, ha='center', va='top',
                            fontsize=8, color=BURST_COLOR)

    loss = result['loss']
    ax.set_title(f"{result['label']}  -  overall loss {loss['loss_pct']:.2f}%  "
                 f"({loss['expected'] - loss['received']}/{loss['expected']} packets)")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Packet loss (%)')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison(results, out_path):
    """Bar chart: overall loss % per condition, exact value on each bar."""
    labels   = [r['label'] for r in results]
    loss_pct = [r['loss']['loss_pct'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, loss_pct, color=BAR_COLOR, width=0.6)

    # Direct value labels - exact numbers on the bars, per request
    for bar, pct in zip(bars, loss_pct):
        ax.annotate(f'{pct:.2f}%',
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Packet loss (%)')
    ax.set_title('SPEAR Wi-Fi link: packet loss by test condition')
    ax.set_ylim(0, max(loss_pct) * 1.2 if max(loss_pct) > 0 else 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)   # grid behind the bars

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    results = []
    for i, (filename, label) in enumerate(LINK_TESTS.items()):
        # Resolve relative names against Data Logs/ the same way load_log does,
        # just to give a clear error before pandas does
        path = filename if os.path.isabs(filename) else \
            os.path.join(script_dir, "Data Logs", filename)
        if not os.path.exists(path):
            print(f"SKIP: {filename} not found (condition '{label}') - "
                  f"edit LINK_TESTS at the top of this script")
            continue

        r = analyze_one(filename, label)
        results.append(r)

        ts_path = os.path.join(out_dir, f"link_timeseries_{i+1}.png")
        plot_timeseries(r, ts_path)

        # Terminal summary per file
        loss    = r['loss']
        gaps    = r['gaps']
        max_gap = int(gaps['gap_size'].max()) if len(gaps) else 0
        print(f"\n{label}")
        print(f"  file:      {os.path.basename(r['filename'])}")
        print(f"  loss:      {loss['loss_pct']:.2f}%  "
              f"({loss['expected'] - loss['received']} of {loss['expected']} packets)")
        print(f"  gaps:      {len(gaps)} total, "
              f"{len(r['bursts'])} bursts (>= {BURST_GAP_MIN} consecutive)")
        print(f"  max gap:   {max_gap} consecutive packets")
        print(f"  plot:      {os.path.relpath(ts_path, script_dir)}")

    if not results:
        print("\nNo files analyzed. Fill in LINK_TESTS with real filenames.")
        return

    bar_path = os.path.join(out_dir, "link_loss_comparison.png")
    plot_comparison(results, bar_path)
    print(f"\nComparison chart: {os.path.relpath(bar_path, script_dir)}")


if __name__ == '__main__':
    main()
