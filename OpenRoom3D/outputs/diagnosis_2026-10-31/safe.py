import torch, numpy as np, pathlib
torch.serialization.add_safe_globals([(np._core.multiarray.scalar,"numpy.core.multiarray.scalar"),np.dtype,type(np.dtype("float64")),type(np.dtype("int64")),pathlib.PosixPath])
