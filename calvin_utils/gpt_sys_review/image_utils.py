import os
import fitz  # PyMuPDF
import json
from pdf2image import convert_from_path
import cv2
import numpy as np

class ImageExtractor:
    """Class to extract figures from PDFs. 
    Able to handle both native and image-based PDFs.

    Attributes:
        filepath (str): Path to the PDF file.
        dir (str): Directory of the PDF file.
        is_native_pdf (bool): Whether the PDF is native or image-based.
        pmid (str): The PMID of the PDF.
        img_paths (list): List of paths to extracted images.
    
    Methods:
        extract_images(): Wrapper function to extract images based on PDF type.
        extract_from_native_pdf(): Extract figures from a native PDF using PyMuPDF.
        handle_image_based_pdf(): Convert image-based PDF pages to images and extract figures.
    """
    def __init__(self, pdf_file_path):
        """Init method for ImageExtractor."""
        self.filepath = pdf_file_path
        self.dir = os.path.dirname(pdf_file_path)
        self.is_native_pdf = False
        self.pmid = None
        self.img_paths = []
        self._check_pdf_type()
        self._determine_pmid()

    def extract_images(self):
        """Wrapper function to extract images based on PDF type."""
        if self.is_native_pdf:
            self.extract_from_native_pdf()
        else:
            self.handle_image_based_pdf()

    def extract_from_native_pdf(self):
        """Extract figures from a native PDF using PyMuPDF."""
        img_counter = 0  # Initialize global image counter
        try:
            with fitz.open(self.filepath) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    for img in page.get_images(full=True):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
                        image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
                        if self._check_valid_img(image):
                            pmid_prefix = f"{self.pmid}_" if self.pmid else ""
                            img_filepath = f"{self.dir}/images/{pmid_prefix}figure_{img_counter}.png"
                            os.makedirs(os.path.dirname(img_filepath), exist_ok=True)
                            with open(img_filepath, "wb") as f:
                                f.write(image_bytes)
                            self._make_empty_json(img_counter)
                            self.img_paths.append(img_filepath)
                            img_counter += 1
        except Exception as e:
            print(f"Error extracting from native PDF: {e}")

    def handle_image_based_pdf(self):
        """Convert image-based PDF pages to images and extract figures."""
        try:
            pages = convert_from_path(self.filepath, 300)  # DPI set to 300 for good quality
            img_counter = 0  # Initialize a counter for extracted images globally

            for page_num, page in enumerate(pages):
                img_filepath = f"{self.dir}/images/page_{page_num}.png"
                os.makedirs(os.path.dirname(img_filepath), exist_ok=True)
                page.save(img_filepath, 'PNG')
                img_counter = self._crop_boxes_in_image(img_filepath, img_counter)
                os.remove(img_filepath)

        except Exception as e:
            print(f"Error handling image-based PDF: {e}")

    def _crop_boxes_in_image(self, img_path, img_counter):
        """Crop boxes (figures) from the given image and save them, filtering out small areas."""
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 1000:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w / float(h) < 0.1 or w / float(h) > 10 or w < 100 or h < 100:
                continue
            cropped_img = img[y:y+h, x:x+w]
            if self._check_valid_img(cropped_img):  # Check if the cropped image is valid
                pmid = f"{self.pmid}_" if self.pmid else ""
                crop_img_path = f"{self.dir}/images/{pmid}figure_{img_counter}.png"
                cv2.imwrite(crop_img_path, cropped_img)
                self._make_empty_json(img_counter)
                self.img_paths.append(crop_img_path)
                img_counter += 1

        return img_counter

    def _determine_pmid(self):
        """Determine the PMID of the PDF based on the filename."""
        # PMID should be the numeric part of the filename
        filename = os.path.basename(self.filepath)
        pmid = "".join([char for char in filename if char.isdigit()])
        if not pmid or len(pmid) < 4:
            self.pmid = None
            return self
        self.pmid = pmid
        return self

    def _check_pdf_type(self):
        """Determine if the PDF is native or image-based by checking for substantial text."""
        try:
            with fitz.open(self.filepath) as doc:
                for page_num in range(min(5, len(doc))):  # Check the first 5 pages
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if len(text) > 50:  # Assuming more than 50 characters indicates native
                        self.is_native_pdf = True
                        break  # Found substantial text, no need to check further
        except Exception as e:
            print(f"Error checking PDF type: {e}")

    def _check_valid_img(self, img):
        """Check if the image has realistic dimensions and characteristics of neuroimaging data."""
        h, w = img.shape[:2]

        aspect_ratio = w / float(h)
        min_width, min_height = 100, 100  # Minimum dimensions
        min_area = 1000  # Minimum pixel area
        max_aspect_ratio = 10  # Maximum aspect ratio
        if not (w >= min_width and h >= min_height and w * h >= min_area and
                0.1 <= aspect_ratio <= max_aspect_ratio):
            return False  # Image fails basic dimension checks

        # Check for unique pixel intensities and variance
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        unique_intensities = len(np.unique(grayscale_img))
        variance = np.var(grayscale_img)

        min_unique_intensities = 100  # Expect at least 100 unique intensities for complexity
        min_variance = 800  # Set based on expected variance in neuroimaging data

        if unique_intensities >= min_unique_intensities and variance >= min_variance:
            return True  # Image passes all checks
        return False  # Image is likely not neuroimaging data or is too simple
    
    def _make_empty_json(self, img_counter):
        """Creates an empty JSON file with a predefined structure."""
        pmid_prefix = f"{self.pmid}_" if self.pmid else ""
        json_filepath = f"{self.dir}/images/{pmid_prefix}figure_{img_counter}.json"
        os.makedirs(os.path.dirname(json_filepath), exist_ok=True)
        json_content = {
            "is_brain": "", "slice_orientation": "", "modality": "", "has_lesion": "",
            "lesion_location": "", "lesion_etiology": "", "mni_slice_index": ""
        }
        with open(json_filepath, "w") as f:
            f.write(json.dumps(json_content, indent=4))