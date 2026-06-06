"""
Hand Gesture Recognition — Webcam Inference
============================================
Optimized for: 10 classes × 200 images dataset

Fixes for high-confidence wrong predictions:
  1. Temperature scaling      → deflates overconfident softmax (main fix)
  2. Entropy rejection        → rejects when model is genuinely uncertain
  3. CLAHE preprocessing      → exact match to training pipeline
  4. MediaPipe hand tracking  → tight ROI, removes background noise
  5. Majority-vote smoothing  → stable label over rolling window
  6. Rich HUD                 → confidence bars for all 10 classes

Requirements:
    pip install opencv-python mediapipe tensorflow numpy
"""

import cv2
import numpy as np
import time
from collections import deque, Counter
from tensorflow.keras.models import load_model

try:
    import mediapipe as mp
    MP = True
except ImportError:
    MP = False
    print("[WARN] mediapipe not found. pip install mediapipe")


# ── CONFIG ────────────────────────────────────────────────────────────────────

IMAGE_SIZE           = 96       # must match training
MODEL_PATH           = "gesture_model.h5"
CLASS_NAMES_PATH     = "class_names.txt"
TEMPERATURE_PATH     = "temperature.txt"

CONFIDENCE_THRESHOLD = 50       # % after calibration — 50% is real with 10 classes
ENTROPY_REJECT       = 0.78     # reject if entropy > 78% of maximum possible
HISTORY_SIZE         = 18       # frames for majority vote
PREDICT_EVERY        = 2        # run model every N frames
BOX_SIZE             = 320      # static box fallback size

# Colours
GREEN  = (0,  230,  80)
RED    = (0,   60, 230)
BLUE   = (255, 140,  0)
YELLOW = (0,  220, 220)
WHITE  = (230, 230, 230)
DARK   = (18,  18,  18)


# ── LOAD ──────────────────────────────────────────────────────────────────────

print("Loading model...")
model = load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH) as f:
    CLASS_NAMES = [l.strip() for l in f if l.strip()]

NUM_CLASSES = len(CLASS_NAMES)
MAX_ENTROPY = np.log(NUM_CLASSES)

try:
    with open(TEMPERATURE_PATH) as f:
        T = float(f.read().strip())
    print(f"Temperature T={T:.2f} loaded")
except FileNotFoundError:
    T = 1.5    # safe default — assumes mild overconfidence
    print(f"[WARN] temperature.txt not found. Using default T={T}")

print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}\n")


# ── MEDIAPIPE ─────────────────────────────────────────────────────────────────

if MP:
    mp_hands   = mp.solutions.hands
    mp_draw    = mp.solutions.drawing_utils
    detector   = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.55
    )


# ── PREPROCESSING (must exactly match training) ───────────────────────────────

def preprocess(roi):
    img   = cv2.resize(roi, (IMAGE_SIZE, IMAGE_SIZE))
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    rgb   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return (rgb / 255.0).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3).astype("float32")


# ── TEMPERATURE SCALING ───────────────────────────────────────────────────────

def calibrate(probs, temperature):
    """Apply temperature scaling to soften overconfident softmax."""
    if temperature == 1.0:
        return probs
    log_p = np.log(probs + 1e-9) / temperature
    log_p -= log_p.max()
    exp_p = np.exp(log_p)
    return exp_p / exp_p.sum()


# ── ENTROPY CHECK ─────────────────────────────────────────────────────────────

def high_entropy(probs):
    H = -np.sum(probs * np.log(probs + 1e-9))
    return H / MAX_ENTROPY > ENTROPY_REJECT


# ── ROI EXTRACTION ────────────────────────────────────────────────────────────

def get_roi(frame, mp_res):
    h, w = frame.shape[:2]
    found = False

    if MP and mp_res and mp_res.multi_hand_landmarks:
        lm = mp_res.multi_hand_landmarks[0]
        xs = [int(p.x * w) for p in lm.landmark]
        ys = [int(p.y * h) for p in lm.landmark]

        pad  = 50
        x1   = max(0,     min(xs) - pad)
        y1   = max(0,     min(ys) - pad)
        x2   = min(w - 1, max(xs) + pad)
        y2   = min(h - 1, max(ys) + pad)

        # Enforce square crop
        side = max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x1 = max(0,     cx - side // 2)
        y1 = max(0,     cy - side // 2)
        x2 = min(w - 1, cx + side // 2)
        y2 = min(h - 1, cy + side // 2)

        mp_draw.draw_landmarks(
            frame, lm, mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,255,180), thickness=1, circle_radius=2),
            mp_draw.DrawingSpec(color=(255,200,0), thickness=1)
        )
        found = True
    else:
        side = BOX_SIZE
        x1 = w // 2 - side // 2
        y1 = h // 2 - side // 2
        x2 = x1 + side
        y2 = y1 + side

    return frame[y1:y2, x1:x2], x1, y1, x2, y2, found


# ── HUD ───────────────────────────────────────────────────────────────────────

def blend_rect(frame, x1, y1, x2, y2, color=DARK, alpha=0.62):
    sub = frame[y1:y2, x1:x2]
    rect = np.full_like(sub, color)
    cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
    frame[y1:y2, x1:x2] = sub

def draw_bar(frame, x, y, label, pct, color, w=195, h=18):
    filled = int(w * min(pct, 100) / 100)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40,40,40), -1)
    cv2.rectangle(frame, (x, y), (x + filled, y + h), color, -1)
    cv2.putText(frame, f"{label}  {pct:.1f}%",
                (x+4, y+13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)

def draw_hud(frame, gesture, conf, all_probs, fps, box, hand_found, uncertain):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box

    # Box colour
    if not hand_found:   box_col = BLUE
    elif uncertain:      box_col = RED
    else:                box_col = GREEN
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 2)

    # ── Top panel ────────────────────────────────────────────────────────────
    blend_rect(frame, 0, 0, W, 100)

    label = "UNCERTAIN — move hand" if uncertain else gesture
    cv2.putText(frame, f"Gesture: {label}", (15, 45),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, box_col, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Confidence: {conf:.1f}%  (calibrated T={T:.1f})",
                (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:.0f}", (W-110, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140,255,140), 1, cv2.LINE_AA)

    badge     = "Hand DETECTED" if hand_found else "Hand NOT FOUND"
    badge_col = GREEN if hand_found else RED
    cv2.putText(frame, badge, (W-210, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_col, 1, cv2.LINE_AA)

    # ── All-class confidence panel (right side) ───────────────────────────────
    pw, ph = 240, NUM_CLASSES * 26 + 35
    px, py = W - pw - 12, 110
    blend_rect(frame, px - 5, py - 5, px + pw + 5, py + ph, alpha=0.65)
    cv2.putText(frame, "All Classes (calibrated):", (px, py + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,180), 1, cv2.LINE_AA)

    # Colour gradient: top class green, others fade to grey
    sorted_idx = np.argsort(all_probs)[::-1]
    rank_color = [GREEN, (0,200,220), (0,160,255), (100,100,255)] + \
                 [(90,90,90)] * (NUM_CLASSES - 4)

    for rank, idx in enumerate(sorted_idx):
        by = py + 28 + rank * 24
        draw_bar(frame, px, by, CLASS_NAMES[idx],
                 all_probs[idx] * 100, rank_color[rank], w=225, h=20)

    # ── Bottom hint ───────────────────────────────────────────────────────────
    blend_rect(frame, 0, H - 30, W, H)
    cv2.putText(frame, "Q = quit  |  S = snapshot",
                (15, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, YELLOW, 1, cv2.LINE_AA)

    return frame


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("ERROR: Webcam not found.")
    exit()

print("Webcam open. Place your hand in frame.")
print(f"Threshold={CONFIDENCE_THRESHOLD}% | T={T:.2f} | {NUM_CLASSES} classes\n")

history       = deque(maxlen=HISTORY_SIZE)
last_gesture  = "Waiting..."
last_conf     = 0.0
last_probs    = np.ones(NUM_CLASSES) / NUM_CLASSES   # uniform start
last_uncertain= False
frame_count   = 0
snap_count    = 0
fps_t         = time.time()
fps_val       = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    now     = time.time()
    fps_val = 1.0 / max(now - fps_t, 1e-6)
    fps_t   = now

    mp_res = None
    if MP:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_res = detector.process(rgb)

    roi, x1, y1, x2, y2, hand = get_roi(frame, mp_res)

    frame_count += 1
    if frame_count % PREDICT_EVERY == 0 and roi.size > 0:
        try:
            raw       = model.predict(preprocess(roi), verbose=0)[0]
            cal       = calibrate(raw, T)           # temperature scaling
            uncertain = high_entropy(cal)            # entropy check
            pred_idx  = int(np.argmax(cal))
            conf      = float(cal[pred_idx]) * 100

            last_probs     = cal
            last_uncertain = uncertain

            if not uncertain and conf >= CONFIDENCE_THRESHOLD:
                history.append(CLASS_NAMES[pred_idx])
                if len(history) >= 4:
                    last_gesture = Counter(history).most_common(1)[0][0]
                else:
                    last_gesture = CLASS_NAMES[pred_idx]
                last_conf = conf
            else:
                history.clear()
                last_gesture = "No Gesture"
                last_conf    = conf

        except Exception as e:
            print(f"[ERR] {e}")

    frame = draw_hud(
        frame, last_gesture, last_conf, last_probs,
        fps_val, (x1, y1, x2, y2), hand, last_uncertain
    )

    cv2.imshow("Hand Gesture Recognition — 10 Classes", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    elif k == ord("s"):
        snap_count += 1
        fname = f"snapshot_{snap_count:03d}.jpg"
        cv2.imwrite(fname, frame)
        print(f"Saved {fname}")

cap.release()
cv2.destroyAllWindows()
if MP:
    detector.close()
print("Done.")