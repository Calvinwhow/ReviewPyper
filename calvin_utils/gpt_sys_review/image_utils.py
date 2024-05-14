import os
import json
import io

import fitz  # PyMuPDF
import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from tqdm import tqdm
from bids import BIDSLayout
from PIL import Image as PILImage
from ipywidgets import VBox, HBox, Image, Label, Button, Layout
from IPython.display import display

rpp_config_path = os.path.join(os.path.dirname(__file__), 'rpp_config.json')

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
        self.is_valid_pdf = False
        self.is_native_pdf = False
        self.pmid = None
        self.img_paths = []
        self._determine_if_valid_pdf() #Sometimes PDFs are corrupted and cannot be opened
        self._check_pdf_type()
        self._determine_pmid()

    def extract_images(self):
        """Wrapper function to extract images based on PDF type."""
        if not self.is_valid_pdf:
            print("PDF is not valid.")
            return self
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
                            pmid_prefix = f"pmid-{self.pmid}_" if self.pmid else ""
                            img_filepath = f"{self.dir}/images/{pmid_prefix}img-{img_counter}.png"
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
                pmid_prefix = f"pmid-{self.pmid}_" if self.pmid else ""
                img_filepath = f"{self.dir}/images/{pmid_prefix}img-{img_counter}.png"
                cv2.imwrite(img_filepath, cropped_img)
                self._make_empty_json(img_counter)
                self.img_paths.append(img_filepath)
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
        pmid_prefix = f"pmid-{self.pmid}_" if self.pmid else ""
        json_filepath = f"{self.dir}/images/{pmid_prefix}img-{img_counter}.json"
        os.makedirs(os.path.dirname(json_filepath), exist_ok=True)
        json_content = {
            "is_brain": "", "slice_orientation": "", "modality": "", "has_lesion": "",
            "lesion_location": "", "lesion_etiology": "", "mni_slice_index": ""
        }
        with open(json_filepath, "w") as f:
            f.write(json.dumps(json_content, indent=4))

    def _determine_if_valid_pdf(self):
        """Check if the PDF is a valid file."""
        try:
            with fitz.open(self.filepath) as doc:
                self.is_valid_pdf = True
                return self
        except Exception as e:
            print(f"Error opening PDF: {e}")
            return self

class ImageLabeler:
    """
    Class to label images as brain or non-brain images. For use in a Jupyter notebook.
    Attributes:
        layout (BIDSLayout): BIDSLayout object for the dataset.
        dataset_path (str): Path to the dataset directory.
        images (list): List of image paths in the dataset.
        brain_images (list): List of brain image paths in the dataset.
        non_brain_images (list): List of non-brain image paths in the dataset.
    Methods:
        label_images(n_images=100, allow_repeats=False): Label images as brain or non-brain images.
        collect_brain_images(dir_path): Collect brain images into a directory as symlinks.
        collect_non_brain_images(dir_path): Collect non-brain images into a directory as symlinks.
    """
    def __init__(self, dataset_path):
        self.layout = BIDSLayout(dataset_path, validate=False, config=[rpp_config_path])
        self.dataset_path = dataset_path

    @property
    def images(self):
        """Get a list of image paths in the dataset."""
        images = self.layout.get(extension='png')
        return [image.path for image in images]
    
    @property
    def brain_images(self):
        """Get a list of brain image paths in the dataset."""
        def check_json_sidecar(img_path):
            json_sidecar = img_path.replace('.png', '.json')
            if os.path.exists(json_sidecar):
                with open(json_sidecar, 'r') as f:
                    metadata = json.load(f)
                return metadata.get('is_brain', False)
            return False
        
        image_list = self.images
        return [img for img in image_list if check_json_sidecar(img)]
        
    @property
    def non_brain_images(self):
        """Get a list of non-brain image paths in the dataset."""
        def check_json_sidecar(img_path):
            json_sidecar = img_path.replace('.png', '.json')
            if os.path.exists(json_sidecar):
                with open(json_sidecar, 'r') as f:
                    metadata = json.load(f)
                # Explicitly check if is_brain is False
                return metadata.get('is_brain', None) is False
            return False

        image_list = self.images
        return [img for img in image_list if check_json_sidecar(img)]
       
    def label_images(self, n_images=100, allow_repeats=False):
        """
        Label images as brain or non-brain images. Displays images in a Jupyter notebook.
        """
        count = 0
        for i, pmid in enumerate(self.layout.get_pubmed_ids()):
            images = self.layout.get(pubmed_id=pmid, extension='png')
            for image in images:
                if count >= n_images:
                    return self
                img_path = image.path

                if allow_repeats:
                    already_labeled = False
                else:
                    already_labeled = self._is_already_labeled(img_path)

                if already_labeled:
                    continue

                image_bytes = self._read_image_as_byte_array(img_path, max_width=500)

                display(VBox([
                    Image(value=image_bytes, format='png', width=500),
                    Label('Is this a brain image?')
                ]))

                yes_button = Button(description='Yes', button_style='success', layout=Layout(width='auto'))
                no_button = Button(description='No', button_style='danger', layout=Layout(width='auto'))
                buttons = [yes_button, no_button]

                for button in buttons:
                    button.on_click(self._on_button_clicked(buttons, img_path, button.description == 'Yes'))

                display(VBox([HBox(buttons), Label('')]))
                count += 1

        return self

    def _is_already_labeled(self, img_path):
        """Checks if an image is already labeled as brain or non-brain."""
        json_sidecar = img_path.replace('.png', '.json')
        if os.path.exists(json_sidecar):
            with open(json_sidecar, 'r') as f:
                metadata = json.load(f)
                label = metadata.get('is_brain', '')
                if label != '':
                    return True
        return False
    
    def collect_brain_images(self, dir_path):
        """Collect brain images into a directory as symlinks."""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for img_path in self.brain_images:
            symlink_path = os.path.join(dir_path, os.path.basename(img_path))
            try:
                os.symlink(img_path, symlink_path)
            except FileExistsError:
                pass  # Skip if the symlink already exists
        return self

    def collect_non_brain_images(self, dir_path):
        """Collect non-brain images into a directory as symlinks."""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for img_path in self.non_brain_images:
            symlink_path = os.path.join(dir_path, os.path.basename(img_path))
            try:
                os.symlink(img_path, symlink_path)
            except FileExistsError:
                pass  # Skip if the symlink already exists
        return self

    def _save_metadata(self, img_path, is_brain):
        """Save metadata to a JSON sidecar file."""
        json_sidecar = img_path.replace('.png', '.json')
        if os.path.exists(json_sidecar):
            with open(json_sidecar, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        metadata['is_brain'] = is_brain
        with open(json_sidecar, 'w') as f:
            json.dump(metadata, f, indent=4)

    def _reset_button_styles(self, buttons, selected_button):
        """Reset button styles to default."""
        for button in buttons:
            if button == selected_button:
                button.style.button_color = 'lightgreen'
            else:
                button.style.button_color = 'lightgray'

    def _on_button_clicked(self, buttons, img_path, is_brain):
        """Callback function for button clicks."""
        def button_clicked(b):
            self._save_metadata(img_path, is_brain)
            self._reset_button_styles(buttons, b)
            print(f"Metadata saved for {img_path}: is_brain={is_brain}")
        return button_clicked

    def _read_image_as_byte_array(self, img_path, max_width=1000):
        """Read image from path and convert to byte array."""
        with open(img_path, 'rb') as img_file:
            img = PILImage.open(img_file)
            if img.mode == 'CMYK':
                img = img.convert('RGB')
            if img.width > max_width:
                ratio = max_width / float(img.width)
                height = int((float(img.height) * float(ratio)))
                img = img.resize((max_width, height), PILImage.LANCZOS)
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='PNG')
            return byte_arr.getvalue()