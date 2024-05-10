import mss
import numpy


class Grabber:
    type = "mss"
    sct = mss.mss()

    def numpy_flip(self, im):
        """ Most efficient Numpy version as of now. """
        frame = numpy.array(im, dtype=numpy.uint8)
        return numpy.flip(frame[:, :, :3], 2)

    def get_image(self, grab_area):
        """
        Make a screenshot of a given area and return it.
        :param grab_area: Format is {"top": 40, "left": 0, "width": 800, "height": 640}
        :return: numpy array
        """
        # noinspection PyTypeChecker
        return self.numpy_flip(self.sct.grab(grab_area))  # return RGB, not BGRA
