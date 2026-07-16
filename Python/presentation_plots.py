"""
Presentation figures for the SPEAR slide deck.

Generates clean PNGs (200 dpi, 16:9, projector-sized fonts) into
plots_deck/. Traces and bars only: no annotations, arrows, or text
callouts anywhere - labels get added in PowerPoint. Bar value labels are
data, not annotation, and are allowed.

All numbers that CAN come from the pipeline DO: each throw is run through
the same analyze_field_log used by analyze_field_throw.py, so these
figures can never drift out of sync with the analysis. The only constants
are physical measurements (tape distances) and historical values that no
longer exist in the code (the pre-fix range errors), each marked with its
provenance.

Usage:
    python presentation_plots.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import spear_analysis as sa

# ---------------------------------------------------------------------------
# Constants that are measurements or history, not analysis output
# ---------------------------------------------------------------------------

# Tape-measured throw distances, meters (field session 2026-07-15)
TAPE_M = {'throw_011': 7.0, 'throw_012': 10.4, 'throw_013': 16.2}

# Range errors BEFORE the v=0 anchor fix, percent. Provenance: summary
# outputs recorded 2026-07-15 with the old strict-quiet v=0 anchor
# (predicted 8.2 / 16.2 / 23.6 m against tape 7.0 / 10.4 / 16.2 m, i.e.
# +17% / +56% / +46%; see Documentation/POST_FIELD_SESSION_CHANGES.md).
# The old anchor no longer exists in the code, so these cannot be
# recomputed.
OLD_RANGE_ERR_PCT = {'throw_011': +17.0, 'throw_012': +56.0,
                     'throw_013': +46.0}

THROWS = ['throw_011', 'throw_012', 'throw_013']

# ---------------------------------------------------------------------------
# Style: one place for everything projector-related
# ---------------------------------------------------------------------------

FIGSIZE = (12.8, 7.2)     # 16:9; at 200 dpi -> 2560 x 1440 px
DPI = 200
COLOR_MAIN = 'steelblue'  # the repo's series color
COLOR_OLD = '#a0a0a0'     # muted gray for "before" / secondary series
COLOR_REF = '#707070'     # reference lines

plt.rcParams.update({
    'figure.figsize': FIGSIZE,
    'axes.titlesize': 20,     # >= 18 required
    'axes.labelsize': 15,     # >= 14 required
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def save(fig, name, out_dir):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------------------
# Run the real pipeline on each throw (no duplicated analysis logic)
# ---------------------------------------------------------------------------

def analyze(name, params, calibration, throws_dir):
    """Load a throw's trimmed record and run the standard pipeline on it.

    Prefers the _trimmed.csv the normal workflow produces (timestamps from
    ~0, which is what a slide should show); creates it via the library's
    own trim function if it does not exist yet. Analysis is
    sa.analyze_field_log, the same call analyze_field_throw.py makes.
    """
    trimmed = os.path.join(throws_dir, f"{name}_trimmed.csv")
    if not os.path.exists(trimmed):
        src = os.path.join(throws_dir, f"{name}.csv")
        df_raw = sa.load_log(src)
        df = sa.apply_bias(df_raw, calibration=calibration)
        phases = sa.detect_phases(df, params)
        trimmed = sa.write_trimmed_copy(df_raw, phases, params, src)

    df = sa.apply_bias(sa.load_log(trimmed), calibration=calibration)
    return sa.analyze_field_log(df, params, log_name=name)


def attitude_checks(res):
    """The two Data-quality attitude numbers, same slices as the summary.

    Whole record = mean truth angle over pre-impact in-gate samples
    (convergence-sensitive); v=0 window = mean over the stillest pre-throw
    stretch only (the attitude that feeds release elevation). Mirrors
    print_field_summary exactly; recomputed here rather than hardcoded so
    the deck cannot disagree with the tool.
    """
    ph = res.phases
    end = ph.impact_idx if ph.impact_idx is not None else len(res.truth_angle)
    whole = res.truth_angle[:end]
    whole = whole[np.isfinite(whole)]
    j0, j1 = ph.v0_window
    vwin = res.truth_angle[j0:j1 + 1]
    vwin = vwin[np.isfinite(vwin)]
    return float(np.mean(whole)), float(np.mean(vwin))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    throws_dir = os.path.join(script_dir, 'Data Logs', 'throws')
    out_dir = os.path.join(script_dir, 'plots_deck')
    os.makedirs(out_dir, exist_ok=True)

    params = sa.SpearParams()
    calibration = sa.load_calibration()

    print("Running pipeline on throws...")
    results = {name: analyze(name, params, calibration, throws_dir)
               for name in THROWS}
    print(f"Writing figures to {os.path.relpath(out_dir, script_dir)}/")

    r13 = results['throw_013']
    t13 = r13.t
    ph13 = r13.phases

    # --- Figure 1: anatomy - |accel| of one throw, trace only ---
    fig, ax = plt.subplots()
    ax.plot(t13, ph13.accel_mag, color=COLOR_MAIN, linewidth=1.2)
    ax.set_title('Throw 013: acceleration magnitude')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|accel| (m/s$^2$)')
    save(fig, 'fig1_anatomy.png', out_dir)

    # --- Figure 2: rolling variance, log scale, trace only ---
    fig, ax = plt.subplots()
    ax.semilogy(t13, ph13.rolling_var, color=COLOR_MAIN, linewidth=1.2)
    ax.set_title('Throw 013: rolling variance of |accel|')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Variance (m/s$^2$)$^2$, log scale')
    ax.grid(True, which='both', alpha=0.3)
    save(fig, 'fig2_variance.png', out_dir)

    # --- Figure 3: integrated speed, v=0 anchor to impact ---
    # res.speed is NaN outside the integration window by design; slicing
    # [v0, impact] plots exactly the span the pipeline trusts.
    i0, i1 = ph13.v0_idx, ph13.impact_idx
    fig, ax = plt.subplots()
    ax.plot(t13[i0:i1 + 1], r13.speed[i0:i1 + 1],
            color=COLOR_MAIN, linewidth=1.6)
    ax.set_title('Throw 013: integrated speed, v=0 anchor to impact')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    save(fig, 'fig3_speed.png', out_dir)

    # --- Shared per-throw pipeline numbers for figures 4-6 ---
    labels = [n.replace('throw_0', 'Throw ') for n in THROWS]
    x = np.arange(len(THROWS))
    new_err = []      # current signed range error, percent (computed)
    predicted = []    # current predicted range, m (computed)
    att_whole = []    # attitude check, whole record (computed)
    att_v0 = []       # attitude check, v=0 window (computed)
    for name in THROWS:
        res = results[name]
        pred = res.ballistic['range_m']
        predicted.append(pred)
        new_err.append((pred - TAPE_M[name]) / TAPE_M[name] * 100.0)
        w, v = attitude_checks(res)
        att_whole.append(w)
        att_v0.append(v)

    # --- Figure 4: range error before/after the v=0 anchor fix ---
    # Bars show ABSOLUTE error (magnitude is the story); the value labels
    # carry the sign, which is data.
    width = 0.38
    fig, ax = plt.subplots()
    b_old = ax.bar(x - width / 2, [abs(OLD_RANGE_ERR_PCT[n]) for n in THROWS],
                   width, color=COLOR_OLD, label='old anchor (strict quiet)')
    b_new = ax.bar(x + width / 2, [abs(e) for e in new_err],
                   width, color=COLOR_MAIN, label='current anchor (pre-throw hold)')
    ax.bar_label(b_old, labels=[f"{OLD_RANGE_ERR_PCT[n]:+.0f}%" for n in THROWS],
                 fontsize=13)
    ax.bar_label(b_new, labels=[f"{e:+.0f}%" for e in new_err], fontsize=13)
    ax.set_title('Range prediction error: v=0 anchor, before and after')
    ax.set_xticks(x, labels)
    ax.set_ylabel('|Range error| (%)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.grid(False, axis='x')
    ax.set_axisbelow(True)
    ax.legend()
    save(fig, 'fig4_v0_before_after.png', out_dir)

    # --- Figure 5: predicted vs tape, y = x reference ---
    tape = [TAPE_M[n] for n in THROWS]
    lim = max(max(tape), max(predicted)) * 1.15
    fig, ax = plt.subplots()
    ax.plot([0, lim], [0, lim], linestyle='--', color=COLOR_REF,
            linewidth=1.2, label='perfect prediction (y = x)')
    ax.scatter(tape, predicted, s=140, color=COLOR_MAIN, zorder=3,
               label='measured throws')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title('Predicted range vs tape measure')
    ax.set_xlabel('Tape-measured range (m)')
    ax.set_ylabel('Predicted range (m)')
    ax.set_aspect('equal')
    ax.legend()
    save(fig, 'fig5_results_vs_tape.png', out_dir)

    # --- Figure 6: attitude metric, whole record vs v=0 window ---
    fig, ax = plt.subplots()
    b_w = ax.bar(x - width / 2, att_whole, width, color=COLOR_OLD,
                 label='whole record (convergence-sensitive)')
    b_v = ax.bar(x + width / 2, att_v0, width, color=COLOR_MAIN,
                 label='v=0 window (feeds release elevation)')
    ax.bar_label(b_w, fmt='%.1f', fontsize=13)
    ax.bar_label(b_v, fmt='%.1f', fontsize=13)
    ax.set_title('Attitude check: mean gravity-direction error')
    ax.set_xticks(x, labels)
    ax.set_ylabel('Mean error (deg)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.grid(False, axis='x')
    ax.set_axisbelow(True)
    ax.legend()
    save(fig, 'fig6_attitude_metric.png', out_dir)

    # --- Figures 7 and 8: the throw_013 diagnostic, one panel per figure ---
    # Same traces as figs 1-2 but WITH the detected-phase overlay (onset,
    # v=0 window and anchor, FLIGHT entry, IMPACT), drawn by the same
    # sa.overlay_phase_markers the field diagnostic uses, so the deck and
    # the tool can never disagree about where a phase boundary sits.
    # Legend upper LEFT per deck layout (the throw lives on the right side
    # of the time axis, so the left is empty space).
    fig, ax = plt.subplots()
    ax.plot(t13, ph13.accel_mag, color=COLOR_MAIN, linewidth=1.2)
    sa.overlay_phase_markers(ax, r13)
    ax.set_title('Throw 013: acceleration magnitude with detected phases')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|accel| (m/s$^2$)')
    ax.legend(loc='upper left')
    save(fig, 'fig7_diagnostic_accel.png', out_dir)

    fig, ax = plt.subplots()
    ax.semilogy(t13, ph13.rolling_var, color=COLOR_MAIN, linewidth=1.2)
    sa.overlay_phase_markers(ax, r13)
    ax.set_title('Throw 013: rolling variance of |accel| with detected phases')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Variance (m/s$^2$)$^2$, log scale')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper left')
    save(fig, 'fig8_diagnostic_variance.png', out_dir)

    # --- Figure 9: fig 7 zoomed to the throw itself ---
    # Same data and overlays, x limited to 8 s onward. The time axis keeps
    # the trimmed record's real timestamps (no re-zeroing): honest labels,
    # directly comparable with fig 7 and the summary output.
    fig, ax = plt.subplots()
    ax.plot(t13, ph13.accel_mag, color=COLOR_MAIN, linewidth=1.4)
    sa.overlay_phase_markers(ax, r13)
    ax.set_xlim(8.0, t13[-1])
    ax.set_title('Throw 013: acceleration magnitude, throw detail')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|accel| (m/s$^2$)')
    ax.legend(loc='upper left')
    save(fig, 'fig9_diagnostic_accel_zoom.png', out_dir)

    print("Done: 9 figures.")


if __name__ == '__main__':
    main()
