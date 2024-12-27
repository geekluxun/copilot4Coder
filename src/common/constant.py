from enum import Enum


class HubOrigin(Enum):
    HF = 'hf'
    HF_MIRROR = 'hf_mirror'
    MS = 'modescope'


if __name__ == '__main__':
    print(HubOrigin.MS.value)
