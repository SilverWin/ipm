"""Image Pattern Matching module
"""

__author__ = "Filip Jankowski"

import cv2

MAX_DIFF = 9999999  # used as maximum difference that is returned when images have nothing in common or are not comparable


class Image:

    def __init__(self, path=None, matrix=None, max_side_length=None):
        """
        Creates normalized image for pattern matching - grayscale and resized if needed
        :param path: path to the image
        :param matrix: opencv image
        :param max_side_length: max length of image's side for resize
        """
        if matrix is None:
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            # check if image is already grey
            if len(matrix.shape) < 3:
                m = matrix.copy()
            else:
                m = cv2.cvtColor(matrix.copy(), cv2.COLOR_BGR2GRAY)

        height, width = m.shape

        if max_side_length is not None:

            if height > width:
                scale = max_side_length / height
            else:
                scale = max_side_length / width

            size = (max(1, int(width * scale)), max(1, int(height * scale)))
            m = cv2.resize(m, size, interpolation=cv2.INTER_AREA)

        self.matrix = m


class ImageDescription:

    def __init__(self, image):
        """
        Creates image description (with opencv descriptors) used for comparing patterns
        :param image: image for which descriptors should be made
        :type image: Image
        """
        orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, WTA_K=2)
        self.key_points, self.descriptors = orb.detectAndCompute(image.matrix, None)

    def compute_difference(self, image_description, match_features, max_feature_difference):
        """
        Computes difference against another image using its image description
        :param image_description: image description for computing difference
        :param match_features: how many key features it should look up for while matching image
        :param max_feature_difference: max difference allowed, key points after above that are ignored
        :return:
        """
        if (len(self.descriptors) <= 0 or len(image_description.descriptors) <= 0
                or len(self.key_points) <= 0 or len(image_description.key_points) <= 0):
            return MAX_DIFF

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.descriptors, image_description.descriptors)

        matches.sort(key=lambda m: m.distance)

        sum = 0
        got = 0

        cnt = min(len(matches), match_features)

        for i in range(cnt):
            m = matches[i]

            if m.distance > max_feature_difference:
                break

            sum = sum + m.distance
            got = got + 1

        if got <= 0:
            return MAX_DIFF

        penalty = 1.0 / (got / match_features)

        return (sum / got) * penalty


class Pattern:

    def __init__(self, image_description, name=None):
        """
        Creates new pattern to be stored in matcher
        :param image_description: image descriptors associated with this pattern
        :type image_description: ImageDescription
        :param name: name of pattern
        """
        self.image_description = image_description
        self.name = name


class Matcher:

    def __init__(self,
                 max_side_length=512,
                 features_required=20,
                 feature_threshold=100,
                 features_average_threshold=100,
                 features_relative_threshold=None):
        """
        Creates new matcher
        :param max_side_length: in what size input images should be processed, None for original size
        :param features_required: how many key features it should look up for while matching image
        :param feature_threshold: threshold for ORB features, lower value = more restrictive, identical images have 0 difference, default value of 100 allows every feature to be included
        :param features_average_threshold: threshold for average value of features found, lower value = more restrictive, identical images have 0 difference
        :param features_relative_threshold: minimum relative threshold between the 1st most similar pattern and 2nd, for example 0.1 requires at least 10% difference between 1st and 2nd pattern), set to None to ignore this threshold
        """
        self._max_side_length = max_side_length
        self.features_required = features_required
        self.feature_threshold = feature_threshold
        self.features_average_threshold = features_average_threshold
        self.features_relative_threshold = features_relative_threshold

        self._patterns = set()

    def add_pattern(self, path=None, matrix=None, name=None):
        """
        Creates new pattern, adds it to matcher and returns created pattern
        :param path: path to the image
        :param matrix: opencv matrix
        :param name: name of pattern
        :return:
        """
        di = ImageDescription(Image(path=path, matrix=matrix, max_side_length=self._max_side_length))
        pattern = Pattern(image_description=di, name=name)
        self._patterns.add(pattern)
        return pattern

    def remove_pattern(self, pattern):
        """
        Removes given pattern from the matcher
        :param pattern: pattern to be removed
        """
        self._patterns.remove(pattern)

    def match(self, path=None, matrix=None, image_description=None):
        """
        Matches image given by path/matrix/image_description against all patterns in this matcher
        :param path: path to image that we will match against
        :param matrix: opencv matrix that we will match against
        :param image_description: image description that we will match against
        :return: list of patterns ordered by difference (min difference first)
        """

        if image_description is None:
            image = Image(path=path, matrix=matrix, max_side_length=self._max_side_length)
            image_description = ImageDescription(image=image)

        diffs = []

        for p in self._patterns:
            d = p.image_description.compute_difference(image_description,
                                                       self.features_required,
                                                       self.feature_threshold)

            if d < self.features_average_threshold:
                diffs.append((p, d))

        diffs.sort(key=lambda t: t[1])

        if len(diffs) >= 2 and self.features_relative_threshold is not None:
            # check if difference in confidence between 1st and 2nd match is enough
            req = diffs[0][1] * (1.0 + self.features_relative_threshold)
            if diffs[1][1] < req:
                return []

        return list(map(lambda t: t[0], diffs))
