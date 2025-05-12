from ._NSpin12 import NSpin12
from ._maps import MapHex127_r3, MapHex127_r4, MapHex133_r3, MapHex156_r3 
from ._gates import gates_from_HH, gates_from_HH2, measure_H_ctm, measure_H_mps

def map_hex(N, r=3):
    if N == 127 and r == 3:
        return MapHex127_r3()
    elif N == 127 and r == 4:
        return MapHex127_r4()
    elif N == 133 and r == 3:
        return MapHex133_r3()
    elif N == 156 and r == 3:
        return MapHex156_r3()
    else:
        raise ValueError()
