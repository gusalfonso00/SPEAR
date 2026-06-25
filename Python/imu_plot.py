# %%
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "Data Logs")

# Find the most recent CSV in Data Logs
csv_files = glob.glob(os.path.join(data_dir, "imu_log_*.csv"))
latest = max(csv_files, key=os.path.getmtime)
print(f"Plotting: {os.path.basename(latest)}")

df = pd.read_csv(latest)
df['t_s'] = df['ms'] / 1000.0

# After loading df, detect gaps larger than 200ms (2x your sample period)
df = df.sort_values('ms').reset_index(drop=True)
dt = df['ms'].diff()
gap_threshold = 200  # ms
gap_indices = df.index[dt > gap_threshold].tolist()

# Insert NaN rows at gap locations so matplotlib breaks the line
for idx in reversed(gap_indices):
    df.loc[idx - 0.5] = [None] * len(df.columns)
df = df.sort_index().reset_index(drop=True)



duration_min = (df['t_s'].iloc[-1] - df['t_s'].iloc[0]) / 60.0
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(df['t_s'], df['temp_C'])
axes[0].set_ylabel('Temp (°C)')

for ax_name in ['ax', 'ay', 'az']:
    axes[1].plot(df['t_s'], df[ax_name], label=ax_name)
axes[1].set_ylabel('Accel (m/s²)')
axes[1].legend()

for ax_name in ['gx', 'gy', 'gz']:
    axes[2].plot(df['t_s'], df[ax_name], label=ax_name)
axes[2].set_ylabel('Gyro (rad/s)')
axes[2].set_xlabel('Time (s)')
axes[2].legend()


fig.text(0.98, 0.98, f'Duration: {duration_min:.1f} min',
         ha='right', va='top', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))


plt.tight_layout()
plt.show()
# %%
