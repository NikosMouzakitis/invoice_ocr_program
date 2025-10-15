import pygame
import sys
import numpy as np
from PIL import Image
import pyperclip
from tkinter import filedialog, Tk
import cv2  # For advanced preprocessing
import pytesseract  # For OCR with Greek support

# Note: You need to install the following packages if not already installed:
# pip install pygame pillow pyperclip numpy opencv-python-headless pytesseract
# Additionally, download and install Tesseract OCR from https://github.com/UB-Mannheim/tesseract/wiki
# Add Tesseract to your system PATH (e.g., C:\Program Files\Tesseract-OCR\tesseract.exe)

# Initialize Pygame
pygame.init()

# Screen dimensions (adjust based on your image size)
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)

class RectSelector:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Image OCR Selector")
	 
        self.clock = pygame.time.Clock()
        self.image = None
        self.scaled_image = None
        self.image_rect = None
        self.image_path = None  # Store path here
        self.rectangles = []  # List of selected rectangles: [(x1, y1, x2, y2, label), ...] IMAGE-RELATIVE coords
        self.current_rect = None
        self.dragging = False
        self.start_pos = None
        self.selected_labels = ['Description', 'Unit', 'Price']  # Fixed order
        self.current_selection_index = 0
        self.extracted_texts = {}  # {label: text}
        self.font = pygame.font.Font(None, 24)
        self.show_results = False
	
        # Fix for Tesseract path (adjust if your install path differs)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    def load_image(self, image_path):
        self.image_path = image_path  # Store the path for OCR cropping
        try:
            pil_image = Image.open(image_path)
            # Scale image to fit screen if too large
            img_width, img_height = pil_image.size
            scale = min(SCREEN_WIDTH / img_width, SCREEN_HEIGHT / img_height)
            new_size = (int(img_width * scale), int(img_height * scale))
            self.scaled_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            self.image = pygame.image.fromstring(self.scaled_image.tobytes(), new_size, self.scaled_image.mode)
            self.image_rect = self.image.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.rectangles = []
            self.current_selection_index = 0
            self.extracted_texts = {}
            self.show_results = False
            print(f"Image loaded: {img_width}x{img_height}, scaled to {new_size}")
        except Exception as e:
            print(f"Error loading image: {e}")

    def get_relative_pos(self, pos):
        """Convert screen pos to image-relative pos"""
        if self.image_rect and self.image_rect.collidepoint(pos):
            rel_x = pos[0] - self.image_rect.left
            rel_y = pos[1] - self.image_rect.top
            return (rel_x, rel_y)
        return None

    def start_drag(self, pos):
        rel_pos = self.get_relative_pos(pos)
        if rel_pos:
            self.start_pos = rel_pos
            self.dragging = True
            self.current_rect = pygame.Rect(rel_pos[0], rel_pos[1], 0, 0)

    def update_drag(self, pos):
        if self.dragging and self.start_pos:
            rel_pos = self.get_relative_pos(pos)
            if rel_pos:
                self.current_rect = pygame.Rect(
                    min(self.start_pos[0], rel_pos[0]),
                    min(self.start_pos[1], rel_pos[1]),
                    abs(rel_pos[0] - self.start_pos[0]),
                    abs(rel_pos[1] - self.start_pos[1])
                )

    def end_drag(self, pos):
        rel_pos = self.get_relative_pos(pos)
        if self.dragging and self.current_rect and self.current_rect.width > 10 and self.current_rect.height > 10:  # Min size check
            # Check for overlap with existing rectangles (all relative)
            for existing_rect in self.rectangles:
                ex1, ey1, ex2, ey2 = existing_rect[:4]
                if (self.current_rect.colliderect(pygame.Rect(ex1, ey1, ex2 - ex1, ey2 - ey1))):
                    print("Overlap detected! Please select non-overlapping areas.")
                    return
            # Add the rectangle (relative coords)
            x1, y1 = self.current_rect.topleft
            x2, y2 = (x1 + self.current_rect.width, y1 + self.current_rect.height)
            label = self.selected_labels[self.current_selection_index]
            self.rectangles.append((x1, y1, x2, y2, label))
            print(f"Selected {label} area: ({x1}, {y1}) to ({x2}, {y2})")  # Relative coords
            self.current_selection_index += 1
            if self.current_selection_index == len(self.selected_labels):
                self.process_ocr()
        self.dragging = False
        self.current_rect = None
        self.start_pos = None

    def preprocess_for_ocr(self, cropped, label):
        """Advanced preprocessing to improve OCR accuracy, especially for Greek text"""
        # cropped is already grayscale PIL Image
        cropped_cv = np.array(cropped)  # Convert to numpy (single channel, uint8)
        
        if label == 'Description':
            # For Greek text: Enhance contrast with CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cropped_cv = clahe.apply(cropped_cv)
            # Denoise
            cropped_cv = cv2.fastNlMeansDenoising(cropped_cv)
        else:
            # For Unit/Price: Simple Otsu thresholding for numbers
            _, cropped_cv = cv2.threshold(cropped_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to PIL
        return Image.fromarray(cropped_cv)

    def process_ocr(self):
        if len(self.rectangles) != 3:
            print("Need exactly 3 selections to process OCR.")
            return

        self.show_results = True
        full_pil_image = Image.open(self.image_path)  # Reload original for accurate cropping
        scale_x = full_pil_image.width / self.scaled_image.width
        scale_y = full_pil_image.height / self.scaled_image.height

        for rect in self.rectangles:
            x1, y1, x2, y2, label = rect
            # Scale back to original coordinates (using relative x1,y1)
            orig_x1 = int(x1 * scale_x)
            orig_y1 = int(y1 * scale_y)
            orig_x2 = int(x2 * scale_x)
            orig_y2 = int(y2 * scale_y)
            # Crop the region
            cropped = full_pil_image.crop((orig_x1, orig_y1, orig_x2, orig_y2))
            # Check crop size
            crop_w, crop_h = cropped.size
            if crop_w < 20 or crop_h < 10:
                print(f"Warning: {label} crop too small ({crop_w}x{crop_h}) - may fail OCR.")
            # Convert to grayscale for better OCR
            cropped_gray = cropped.convert('L')
            # Preprocess based on label
            processed_cropped = self.preprocess_for_ocr(cropped_gray, label)
            # OCR configs
            if label == 'Description':
                lang = 'ell'  # Greek
                config = '--psm 6'  # Assume uniform block of text
            else:
                lang = 'eng'  # English for numbers/units
                whitelist = '0123456789.,x/€$% '
                config = f'--psm 7 -c tessedit_char_whitelist={whitelist}'  # Single text line
            
            text = pytesseract.image_to_string(processed_cropped, lang=lang, config=config).strip()
            if not text:
                text = "No text detected - try larger/tighter selection"
            self.extracted_texts[label] = text
            print(f"{label}: {text}")

    def draw(self):
        self.screen.fill(WHITE)
        if self.image:
            self.screen.blit(self.image, self.image_rect)

        # Draw selected rectangles (convert relative to screen-absolute)
        img_offset = self.image_rect.topleft
        for rect in self.rectangles:
            x1, y1, x2, y2, label = rect
            abs_rect = pygame.Rect(x1 + img_offset[0], y1 + img_offset[1], x2 - x1, y2 - y1)
            color = RED if label == 'Description' else GREEN if label == 'Unit' else BLUE
            pygame.draw.rect(self.screen, color, abs_rect, 2)
            text_surf = self.font.render(label, True, color)
            self.screen.blit(text_surf, (abs_rect.left, abs_rect.top - 20))

        # Draw current drag rectangle (relative to screen)
        if self.current_rect:
            abs_current = pygame.Rect(self.current_rect.left + img_offset[0], self.current_rect.top + img_offset[1],
                                      self.current_rect.width, self.current_rect.height)
            pygame.draw.rect(self.screen, BLACK, abs_current, 2)

        # Instructions
        if not self.show_results:
            instr_text = [
                "Drag to select areas (Description -> Unit -> Price)",
                "Select non-overlapping rectangles on the same row.",
                f"Selection {self.current_selection_index + 1}/3: {self.selected_labels[self.current_selection_index] if self.current_selection_index < len(self.selected_labels) else 'Done'}",
                "Press 'R' to reset selections.",
                "After 3 selections, OCR will auto-process."
            ]
            y_offset = 10
            for text in instr_text:
                text_surf = self.font.render(text, True, BLACK)
                self.screen.blit(text_surf, (10, y_offset))
                y_offset += 25
        else:
            # Show extracted texts as clickable "buttons"
            y_offset = 10
            result_text = "Extracted Texts (Click to copy the VALUE to clipboard):"
            text_surf = self.font.render(result_text, True, BLACK)
            self.screen.blit(text_surf, (10, y_offset))
            y_offset += 30
            for label, text in self.extracted_texts.items():
                display_text = f"{label}: {text}"
                text_surf = self.font.render(display_text, True, BLACK)
                self.screen.blit(text_surf, (10, y_offset))
                # For click detection, we'll use approximate rects in handle_click
                y_offset += 25

        pygame.display.flip()

    def handle_click_on_text(self, pos):
        if not self.show_results:
            return
        y_offset = 40  # After title + spacing
        for label, text in self.extracted_texts.items():
            # Approximate rect: assume font height ~20, width up to 500
            text_rect = pygame.Rect(10, y_offset, 500, 25)
            if text_rect.collidepoint(pos):
                pyperclip.copy(text)
                print(f"Copied '{text}' to clipboard.")
                return
            y_offset += 25

    def run(self):
        # Hide Tkinter root window
        root = Tk()
        root.withdraw()  # This hides the main window

        image_path = filedialog.askopenfilename(
            title="Select Invoice Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        root.destroy()  # Clean up

        if not image_path:
            print("No image selected. Exiting.")
            return

        self.load_image(image_path)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.show_results:
                        self.start_drag(event.pos)
                    else:
                        self.handle_click_on_text(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if self.dragging:
                        self.end_drag(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        self.update_drag(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.rectangles = []
                        self.current_selection_index = 0
                        self.show_results = False
                        self.extracted_texts = {}
                        print("Reset selections.")
                    elif event.key == pygame.K_p and len(self.rectangles) == 3 and not self.show_results:
                        self.process_ocr()

            self.draw()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = RectSelector()
    app.run()