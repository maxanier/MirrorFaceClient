import constants as const
import config
import glob,os
import cv2
import picam
import face
import fnmatch
import numpy as np

#Store the models for all faces
models = {}
camera = 0


def init():
    print("Loading training data...")
    os.chdir(config.MODEL_DIR)
    global models
    global camera
    for file in glob.glob("*.xml"):
        print("Loading model {0}".format(file))
        name=file.replace(".xml","")
        models[name]=cv2.face.createEigenFaceRecognizer()
        models[name].load(file)
    print("Training data loaded!")
    camera=picam.OpenCVCapture()


def walk_files(directory, match='*'):
    """Generator function to iterate through all files in a directory recursively
    which match the given filename match parameter.
    """
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, match):
            yield os.path.join(root, filename)


def walk_dirs_exlude(directory, match='*'):
    """Generator function to interate through all subdirectories which do NOT match the given match parameter"""
    for root, dirs, files in os.walk(directory):
        matches = fnmatch.filter(dirs,match)
        for dir in set(dirs).difference(matches):
            yield os.path.join(root,dir)


def prepare_image(filename):
    """Read an image as grayscale and resize it to the appropriate size for
    training the face recognition model.
    """
    return face.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))


def normalize(X, low, high, dtype=None):
    """Normalizes a given array in X to a value between low and high.
    Adapted from python OpenCV face recognition example at:
      https://github.com/Itseez/opencv/blob/2.4/samples/python2/facerec_demo.py
    """
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)


def check():
    """Checks if it can recognize a face. Returns name is successful and None otherwise"""
    print("Looking for face...")
    image = camera.read()
    image = cv2.cvtColor(image)
    # Get coordinates of single face in captured image.
    result = face.detect_single(image)
    if result is None:
        print("Could not detect face!")
        return
    x, y, w, h = result
    # Crop and resize image to face.
    crop = face.resize(face.crop(image, x, y, w, h))
    # Test face against models.
    prob = {}
    for file, model in models.items():
        label, confidence = model.predict(crop)
        print('Predicted {0} face with confidence {1} for {2}'.format(
            'POSITIVE' if label == config.POSITIVE_LABEL else 'NEGATIVE', confidence, file))
        if label == config.POSITIVE_LABEL:
            prob[file] = confidence
    name = None
    conf = 10000000
    for file, confidence in prob.items():
        print("Checking {0} with {1}".format(file, confidence))
        if confidence < conf:
            conf = confidence
            name = file
    return name

def capture_positive(num,name):
    """Capture one positive image. Return 1 if successful and 0 otherwise"""
    path=config.POSITIVE_DIR+"/"+name
    print('Capturing image...')
    image = camera.read()
    # Convert image to grayscale.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Get coordinates of single face in captured image.
    result = face.detect_single(image)
    if result is None:
        print('Could not detect single face!')
        return 0
    x, y, w, h = result
    # Crop image as close as possible to desired face aspect ratio.
    # Might be smaller if face is near edge of image.
    crop = face.crop(image, x, y, w, h)
    # Save image to file.
    filename = os.path.join(path, const.POSITIVE_FILE_PREFIX + '%03d.pgm' % num)
    cv2.imwrite(filename, crop)
    print('Found face and wrote training image', filename)
    return 1

def train(name):
    print("Reading training images for {0}".format(name))
    faces = []
    labels = []
    pos_count = 0
    neg_count = 0
    # Read all positive images
    for filename in walk_files(config.POSITIVE_DIR + "/" + name, '*.pgm'):
        faces.append(prepare_image(filename))
        labels.append(const.POSITIVE_LABEL)
        pos_count += 1
    # Read all negative images
    for filename in walk_files(config.NEGATIVE_DIR, '*.pgm'):
        faces.append(prepare_image(filename))
        labels.append(const.NEGATIVE_LABEL)
        neg_count += 1
    # Read all other positive images as negative
    for dir in walk_dirs_exlude(config.POSITIVE_DIR,name):
        for filename in walk_files(dir ,'*.pgm'):
            faces.append(prepare_image(filename))
            labels.append(const.NEGATIVE_LABEL)
            neg_count += 1

    print('Read', pos_count, 'positive images and', neg_count, 'negative images.')

    # Train model
    print('Training model...')
    model = cv2.face.createEigenFaceRecognizer()
    model.train(np.asarray(faces), np.asarray(labels))

    # Save model results
    model.save(config.MODEL_DIR + "/" + name + ".xml")
    print('Training data saved to', config.MODEL_DIR + "/" + name + ".xml")

    # Save mean and eignface images which summarize the face recognition model.
    mean = model.getMean().reshape(faces[0].shape)
    cv2.imwrite(const.MEAN_FILE, normalize(mean, 0, 255, dtype=np.uint8))
    eigenvectors = model.getEigenVectors()
    pos_eigenvector = eigenvectors[:, 0].reshape(faces[0].shape)
    cv2.imwrite(const.POSITIVE_EIGENFACE_FILE, normalize(pos_eigenvector, 0, 255, dtype=np.uint8))
    neg_eigenvector = eigenvectors[:, 1].reshape(faces[0].shape)
    cv2.imwrite(const.NEGATIVE_EIGENFACE_FILE, normalize(neg_eigenvector, 0, 255, dtype=np.uint8))