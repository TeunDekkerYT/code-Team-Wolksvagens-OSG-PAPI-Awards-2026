import cv2
import numpy as np
import time
from picamera2 import Picamera2
import pygame

def find_pupil_center_fast(gray):
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)   # stronger blur

    # Adaptive‑like threshold via Otsu + inversion
    _, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove tiny noise with morphology
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, thresh

    # Keep only contours that are near the center of the ROI
    # and within a reasonable area window
    center_y, center_x = gray.shape[0] // 2, gray.shape[1] // 2
    min_area, max_area = 30, 800   # tweak to your scale
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        # Compute center of this contour
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Distance from center of ROI
        dist = np.hypot(cx - center_x, cy - center_y)
        if dist > max(gray.shape) * 0.3:  # too far from center
            continue
        candidates.append((cnt, area, cx, cy))

    if not candidates:
        return None, thresh

    # Pick the largest valid contour (near center, correct area)
    largest = max(candidates, key=lambda x: x[1])
    contour, area, cx, cy = largest

    # Fit circle
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    return center, thresh


def main():
    # --- AUDIO INITIALISEREN ---
    pygame.mixer.init()
    pygame.mixer.music.load("beep.mp3")  # zorg dat dit bestaat

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}   # LOWER resolution = more CPU time per frame
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    prev_time = time.time()
    last_seen_time = time.time()
    AWAY_THRESHOLD = 2.0  # in seconds

    frame_count = 0
    SKIP_DETECTION_EVERY = 6   # 60 FPS / 6 ≈ 10 detection frames per second

    alarm_active = False

    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            h, w = frame.shape[:2]

            # Make ROI slightly smaller and centered (eye‑like region)
            roi_w, roi_h = int(w * 0.5), int(h * 0.5)
            x0 = w // 2 - roi_w // 2
            y0 = h // 2 - roi_h // 2
            roi = frame[y0:y0 + roi_h, x0:x0 + roi_w]

            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

            center = None
            thresh = np.zeros_like(gray)

            # Do detection rarely to let CPU focus on quality
            if frame_count % SKIP_DETECTION_EVERY == 0:
                center, thresh = find_pupil_center_fast(gray)

            # --- Alarm logic ---
            if center is not None:
                # Pupil seen: update last_seen_time and silence alarm
                last_seen_time = time.time()
                if alarm_active and pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                alarm_active = False
            else:
                # Only count “away” if detection actually ran on this frame
                if frame_count % SKIP_DETECTION_EVERY == 0:
                    away_time = time.time() - last_seen_time
                    if away_time > AWAY_THRESHOLD:
                        if not alarm_active:
                            pygame.mixer.music.play()
                            alarm_active = True

            # --- Visualization ---
            roi_vis = roi.copy()
            if center is not None:
                cv2.circle(roi_vis, center, 5, (0, 255, 0), -1)
                global_center = (x0 + center[0], y0 + center[1])
                cv2.circle(frame, global_center, 6, (0, 0, 255), -1)

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


if __name__ == "__main__":
    main()
