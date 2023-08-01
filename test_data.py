import os

from matplotlib import pyplot as plt
from pytz import timezone

from mistdata import LSTBinnedTherm, LSTBinnedSpectra

tz = timezone('Canada/Central')
LSTH = 23.9344696
LON = -90.99885
datadir1 = "../data/CSA/mini2/"
datadir2 = "../data/CSA/mini2_hdf/"
datadir = datadir2
names = [f for f in sorted(os.listdir(datadir)) if f.startswith("mini2")]
files = [os.path.join(datadir, f) for f in names]

# print("Loading spectra")
# data = [MISTData.load(f) for f in files][:len(files)//2]
# data = sorted(data, key=lambda x: x.spec.time_start)
# print("Summing spectra")
# datafile = MISTData.from_list(data)
# datafile.save("mist2022half")

# spec = MISTData.load("mist2022full.mist").spec
# therm = spec.therm
# therm.to_lst(LON, tz, inplace=True)
# btherm = LSTBinnedTherm(therm)
# btherm.save("mist2022binnedtherm.bmist")

bin_data = LSTBinnedSpectra.load("models/mist2022binned.bmist")
btherm = LSTBinnedTherm.load("models/mist2022binnedtherm.bmist")
# plt.savefig("mist_data_occupancy.png", dpi=500, bbox_inches='tight')
# btherm.plot_bin(btherm.ambient) #lst 0
# plt.show()
bin_data.plot_bin(0, xlim=(30, 100), percent=True)
plt.show()
# plt.savefig("mist_data_percent_example.png", dpi=300, bbox_inches="tight")
# freq = 40
# ndays = bin_data.temp.shape[0]
# freq_ind = np.searchsorted(bin_data.freq, freq)
# data_freq = bin_data.temp[:, :, freq_ind].flatten()
# time_axis = np.concatenate([bin_data.bin_time[:-1] + LSTH * i for i in range(ndays)])
# plt.plot(time_axis, data_freq)
# plt.show()

# bin_data.plot_bin(0, xlim=(30, 100), percent=True)
# plt.show()

# bin_data.plot_occupancy()
# lst = np.arange(0, 24)
#     bin_data.plot_bin(lst[i], xlim=(30, 100), ylim=(-80, 40))
#     plt.savefig(f"pics/bin_{sr(
# # for i in tqdm(range(len(lst)t)):i).zfill(3)}.png", dpi=500, bbox_inches='tight')
#     plt.clf()
