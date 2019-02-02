try:
    from PIL import Image
except ImportError:
    import Image

from rvl_cdp.config import TESSERACT_PATH
import pytesseract

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


class Tesseract:
    def get_text(self, image):
        return pytesseract.image_to_string(image)

