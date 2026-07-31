from enum import IntEnum
from math import exp
import torch.nn as nn
def ms2h(ms: int) -> float:
    return ms/3600000.0

def h2ms(h: float) -> int:
    return h*3600000.0

def s2h(s: float) -> float:
    return s/3600.0

def h2s(h: float) -> float:
    return h*3600

class StateContent(IntEnum):
    SOLAR                           = 1<<0
    MONTH                           = 1<<1
    HOUR                            = 1<<2
    DAY                             = 1<<3
    NEXT_DAY                        = 1<<4
    HUMIDITY                        = 1<<5
    PRESSURE                        = 1<<6
    CLOUD                           = 1<<7
    SUN_REAL_PREDICTION             = 1<<8
    SUN_ESTIMATE_PREDICTION         = 1<<9
    SUN_ESTIMATE_SINGLE_PREDICTION  = 1<<10
    MINUTE                          = 1<<11
    HOUR_MINUTE                     = 1<<12
    BUFFER                          = 1<<13
    SUNSET_TIME                     = 1<<14
    DAY_AVG                         = 1<<15
    EMBEDDED_CURRENT_DAY            = 1<<16
    EMBEDDED_NEXT_DAY               = 1<<17
    QUANTIZED_DAY                   = 1<<18
    QUANTIZED_PREV_DAY              = 1<<19
    EMBEDDED_PREV_NEXT_DAY          = 1<<20
    IMAGES                          = 1<<21
    SOLAR_HORIZON                   = 1<<22
    NN_PREDICTION                   = 1<<23


def emphasize_diff_sigmoid(x, sharpness=10):
    return 1 / (1 + exp(-sharpness * (x - 0.5)))
