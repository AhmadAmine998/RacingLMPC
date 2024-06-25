from utils import ListDict, pltUtils
import numpy as np

test = np.load('./results/LMPC_CUSTOM1/lmpc_data_20240609-223627.npz',allow_pickle=True)

save_dir = './results/lmppi_map_ind55_n_steps14/'
rec = ListDict()
rec.init('safe_set', 'safe_set_xy', 'mppi_state_records', 'mppi_cov_records', 'control_records')
rec.load('safe_set', 'safe_set_xy', 'mppi_state_records', 'mppi_cov_records', 'control_records', save_dir=save_dir)

track_dir = './results/maps/custom1/'
track = ListDict()
track.init('track_center', 'track_inner', 'track_outer')
track.load('track_center', 'track_inner', 'track_outer', save_dir=track_dir)
track.track_center = track.track_center[0]
track.track_inner = track.track_inner[0]
track.track_outer = track.track_outer[0]

traj = rec.safe_set_xy[-1]
plt_utils = pltUtils()
axs = plt_utils.get_fig()
axs[0].scatter(traj[:, 0]-8.5, traj[:, 1], c=traj[:, 3], cmap='viridis')
axs[0].set_title('Last Optimal Trajectory')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].plot(track.track_center[:, 0], track.track_center[:, 1], 'k')
axs[0].plot(track.track_inner[:, 0], track.track_inner[:, 1], 'b')
axs[0].plot(track.track_outer[:, 0], track.track_outer[:, 1], 'b')
axs[0].plot(test['SS_glob'][-1][0:test['LapTime'][-1], 4], test['SS_glob'][-1][0:test['LapTime'][-1], 5]-8.5, '-r')
plt_utils.colorbar(0)
plt_utils.show()