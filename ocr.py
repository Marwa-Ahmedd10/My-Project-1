import cv2
import pytesseract
import numpy as np
import re
from typing import Optional, Tuple

class OCRModule:
    def __init__(self, tesseract_path: str = r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.debug_mode = False
        self.last_text = ""

    def set_debug(self, enabled: bool) -> None:
        self.debug_mode = enabled

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Equalize histogram to boost contrast
        gray = cv2.equalizeHist(gray)

        # Gaussian blur and Otsu thresholding
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Slight morphological opening to clean noise
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        if self.debug_mode:
            cv2.imshow("OCR Preprocessing", cleaned)
            cv2.waitKey(1)

        return cleaned

    def extract_text(self, image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> str:
        try:
            if roi is not None:
                x, y, w, h = roi
                image = image[y:y+h, x:x+w]
                if image.size == 0:
                    return ""

            processed = self.preprocess_image(image)
            raw_text = pytesseract.image_to_string(
                processed,
                config='--psm 6',
                lang='eng'
            )

            # Clean noisy characters
            cleaned_text = re.sub(r'[^A-Za-z0-9 .,!?\'"\-\(\)\[\]:]', '', raw_text)
            cleaned_text = cleaned_text.strip()

            if cleaned_text and cleaned_text != self.last_text:
                self.last_text = cleaned_text

            if self.debug_mode:
                print(f"OCR Output: {repr(cleaned_text)}")
                debug_img = image.copy()
                cv2.putText(debug_img, f"Text: {cleaned_text[:30]}...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("OCR Debug", debug_img)
                cv2.waitKey(1)

            return cleaned_text

        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return ""

    def draw_roi(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "OCR Region", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return frame

# Standalone test
if __name__ == "__main__":
    ocr = OCRModule()
    ocr.set_debug(True)

    cap = cv2.VideoCapture(0)
    roi = (300, 200, 680, 320)

    frame_count = 0
    cooldown = 30  # Check OCR every 30 frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = ocr.draw_roi(frame, roi)

        if frame_count % cooldown == 0:
            try:
                text = ocr.extract_text(frame, roi)
                if text:
                    print(" OCR Output:", text)
            except Exception as e:
                print(" OCR CRASH:", str(e))

        cv2.imshow("Live OCR", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

