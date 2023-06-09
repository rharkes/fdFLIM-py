"""
test features of the fdflim package
"""
from pathlib import Path
from flifile import FliFile
from fdflim import Reference, Sample
from matplotlib import pyplot as plt

tau_ref = 3.93e-9
frequency = 40e6
ref = Path(
    "G:\\",
    "SurfDrive",
    "Data",
    "2019",
    "04",
    "29",
    "2019-04-10_14.25.59_reference_R6G.fli",
)
sam = Path(
    "G:\\",
    "SurfDrive",
    "Data",
    "2019",
    "04",
    "29",
    "2019-04-09_15.51.21_sample_IsoP_Fors.fli",
)
ref = FliFile(ref)
sam = FliFile(sam)
refdat = ref.getdata()
samdat = sam.getdata()

my_ref = Reference(refdat, tau_ref, frequency, axis=2)
my_sam = Sample(samdat, my_ref)
tau_phi = my_sam.getlifetimephase()
plt.figure()
plt.imshow(tau_phi[:, :, 1], vmin=0, vmax=4e-9)
plt.colorbar()
plt.show()

tau_mod = my_sam.getlifetimemod()
plt.figure()
plt.imshow(tau_mod[:, :, 1], vmin=5e-9, vmax=10e-9)
plt.colorbar()
plt.show()
