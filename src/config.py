import constants as const

state=const.STATUS_RUNNING
# Directories which contain the positive and negative training image data.
POSITIVE_DIR = './training/positive'
NEGATIVE_DIR = './training/negative'
POSITIVE_THRESHOLD = 2000.0
MODEL_DIR = './models'

debug=0
DEBUG_IMAGE= 'capture.pgm'