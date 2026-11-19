"""Deep factor retention with a shared complex moment cache; approximate after compression."""
from .native_structured_pnr import NativeStructuredPNRSampler

class NativeDeepPNRSampler(NativeStructuredPNRSampler):
    def __init__(self,*args,retained=3,**kwargs):
        if isinstance(retained,bool) or not isinstance(retained,int) or not 0<=retained<=7:
            raise ValueError('retained must be an integer in [0,7]')
        # Reuse setup and ABI validation; the new binary accepts the larger depth.
        super().__init__(*args,retained=0,**kwargs)
        self.retained=retained
