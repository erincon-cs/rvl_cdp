try:
    from PIL import Image
except ImportError:
    import Image

from rvl_cdp.config import TESSERACT_PATH
import pytesseract


class Tesseract:
    def __init__(self, path=None):
        if path is None:
            path = TESSERACT_PATH

        pytesseract.pytesseract.tesseract_cmd = path

    def get_text(self, image):
        return pytesseract.image_to_string(image)
