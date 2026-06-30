# Face the AI: Emotion Recognition Challenge

A webcam game for high school students that turns computer vision and machine
learning into a round-based challenge. The player acts out an emotion, the AI
predicts what it sees, and the scoreboard shows who won the round.

The goal is not to hide the AI behind a flashy demo. The goal is to make the
pipeline visible and easy to explain: camera input, facial landmarks, feature
extraction, model training, prediction, scoring, and levels.

## How it works

1. `mediapipe_compat.py` uses MediaPipe Face Landmarker to detect facial
   landmarks from webcam frames.
2. `feature_extraction.py` turns those landmarks into 9 geometry features.
3. `data_collector.py` records labeled samples for three emotions: happy, sad,
   and neutral.
4. `train_model.py` trains a RandomForest classifier on the collected samples.
5. `game.py` runs the live challenge, shows the AI guess, awards points, and
   increases the level as the round count rises.

## What changed in this remake

- The project is now framed as `Face the AI: Emotion Recognition Challenge`.
- The game mechanic is now "player acts, AI guesses" instead of a generic
  expression-matching prototype.
- The supported emotions are now happy, sad, and neutral.
- Old prototype labels like laughing, angry, and confused are ignored during
  training so the project stays aligned with the new concept.

## Project structure

```
Face Expression Classifier/
├── challenge_config.py     # shared labels, paths, and pacing settings
├── mediapipe_compat.py     # MediaPipe Face Landmarker compatibility layer
├── feature_extraction.py   # landmark -> feature vector logic
├── data_collector.py       # STEP 1: collect labeled samples
├── train_model.py          # STEP 2: train + save the classifier
├── game.py                 # STEP 3: play the challenge
├── requirements.txt
├── data/                   # dataset.csv and face_landmarker.task live here
└── models/                 # trained model file goes here
```

## Setup

```powershell
cd path\to\Face Expression Classifier
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Recommended workflow

### 1. Collect samples

```powershell
python data_collector.py
```

Capture about 40-60 samples for each emotion. Press `SPACE` to save a sample,
`N` to move to the next emotion, and `Q` to quit.

### 2. Train the model

```powershell
python train_model.py
```

The trainer ignores old prototype labels and only uses the challenge labels.
If you want a clean rebuild, delete `data/dataset.csv` first and recollect the
three target emotions.

### 3. Play the game

```powershell
python game.py
```

Each round the AI shows an emotion prompt, watches your face in real time,
and scores itself if it predicts correctly before the round ends. If it misses,
the player gets the point. The interface shows the current guess, confidence,
score, level, and round timer.

## Notes

- The model is still a classic machine learning pipeline, not a deep neural
  network. That makes it easier to explain in class.
- The project runs on CPU and does not need a GPU.
- If the camera opens but no face is detected, check lighting and make sure no
  other app is using the webcam.
