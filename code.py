import cv2 
import numpy as np 
import time 
from picamera2 import Picamera2 
import pygame  # nieuw 
__name__ = "__main__" 
def find_pupil_center_fast(gray): 
     blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
     _, thresh = cv2.threshold( 
         blurred, 0, 255, 
         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU 
     ) 
     kernel = np.ones((3, 3), np.uint8) 
     thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1) 
     contours, _ = cv2.findContours( 
         thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE 
     ) 

  

    if not contours: 
         return None, thresh 

  

    largest = max(contours, key=cv2.contourArea) 
     area = cv2.contourArea(largest) 

  

    if area < 30 or area > 2000: 
         return None, thresh 

  

    (x, y), radius = cv2.minEnclosingCircle(largest) 
     center = (int(x), int(y)) 
     return center, thresh 

  

def main(): 
     # --- AUDIO INITIALISEREN --- 
     pygame.mixer.init() 
     pygame.mixer.music.load("beep.mp3")  # zorg dat dit bestand bestaat 

  

    picam2 = Picamera2() 
     config = picam2.create_preview_configuration( 
         main={"size": (320 * 3, 240 * 3), "format": "RGB888"} 
     ) 
     picam2.configure(config) 
     picam2.start() 
     time.sleep(1) 

  

    prev_time = time.time() 

  

    last_seen_time = time.time() 
     AWAY_THRESHOLD = 2.0  # seconden te lang wegkijken 
 

  

    frame_count = 0 
     SKIP_DETECTION_EVERY = 2 

  

    # om te voorkomen dat het mp3'tje continu triggert 
     alarm_active = False 

  

    try: 
         while True: 
             frame = picam2.capture_array() 
             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 

  

            h, w = frame.shape[:2] 
             roi_w, roi_h = int(w * 0.6), int(h * 0.6) 
             x0 = w // 2 - roi_w // 2 
             y0 = h // 2 - roi_h // 2 
             roi = frame[y0:y0 + roi_h, x0:x0 + roi_w] 

  

            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) 

  

            center = None 
             thresh = np.zeros_like(gray) 

  

            if frame_count % SKIP_DETECTION_EVERY == 0: 
                 center, thresh = find_pupil_center_fast(gray) 

  

                if center is not None: 
                     last_seen_time = time.time() 
                     # als we weer kijken, alarm stoppen 
                     if alarm_active and pygame.mixer.music.get_busy(): 
                         pygame.mixer.music.stop() 
                     alarm_active = False 
                 else: 
                     away_time = time.time() - last_seen_time 
                     if away_time > AWAY_THRESHOLD: 
                         if not alarm_active: 
                             pygame.mixer.music.play() 
                             alarm_active = True 

  

            roi_vis = roi.copy() 
             if center is not None: 
                 cv2.circle(roi_vis, center, 4, (0, 255, 0), -1) 
                 global_center = (x0 + center[0], y0 + center[1]) 
                 cv2.circle(frame, global_center, 5, (0, 0, 255), -1) 

  

            current_time = time.time() 
             fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0 
             prev_time = current_time 
             cv2.putText( 
                 frame, f"FPS: {fps:.1f}", (5, 20), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1 
             ) 

  

            cv2.rectangle( 
                 frame, (x0, y0), (x0 + roi_w, y0 + roi_h), 
                 (255, 0, 0), 1 
             ) 

  

            cv2.imshow("Eye ROI + pupil", roi_vis) 
             cv2.imshow("Eye tracker", frame) 
             cv2.imshow("Threshold", cv2.resize(thresh, (160, 120))) 

  

            frame_count += 1 

  

            key = cv2.waitKey(1) & 0xFF 
             if key == ord('q'): 
                 break 

  

    finally: 
         picam2.stop() 
         cv2.destroyAllWindows() 
         pygame.mixer.quit() 

  

  

main() 
