# fdFLIM
Python code for working with frequency domain FLIM data

## Requirements
* Numpy

## Example using flifiles
```
from flifile import FliFile
from fdflim import Sample, Reference
from pathlib import Path
from matplotlib import pyplot as plt

base_path = Path('E:\\', '2020', '07', '15')

# some basic analysis
ref_file = FliFile(Path(base_path, '2020-07-15_14.00.17_reference_test.fli'))
sam_file = FliFile(Path(base_path, '2020-07-15_14.00.22_sample_test.fli'))
ref = Reference(ref_file.getdata(), 3.93E-9, 40E6)
sam = Sample(sam_file.getdata(), ref)
tp = sam.getlifetimephase()
plt.imshow(tp)
plt.show()
```
