try:
    # Python 2 forward compatibility
    range = xrange
except NameError:
    pass


from trustscore.trustscore import TrustScore
from trustscore.trustscore import KNNConfidence
