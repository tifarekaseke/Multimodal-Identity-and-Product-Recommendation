# Multimodal Identity and Product Recommendation

## System integration

The command-line app chains the three saved models. A product category is shown only when both of these are true:

1. The face model recognizes an enrolled person (colombe, nelly, sahra, or tifare).
2. The voice model confirms that the same person spoke.

A recognized face plus another member's voice is denied. An unrecognized face stops the flow before the voice check and before the product model. The product model reads the customer's social profile only. It does not use purchase amount, rating, or a product the customer already bought.

Customer `A178` is the demo visitor. That profile is in the social data and has no matched transaction, so the prediction is for a visit rather than a purchase already on file. `A178` and `178` are the same customer.

### Run the scripted checks

From the repository root:

```
pip install -r requirements.txt
python src/system_cli.py --scenario all --save
python -m unittest discover -s tests -v
```

`--scenario all` runs four checks:

- authorized: colombe's enrolled face and colombe's enrolled voice, then a prediction for customer 178
- unknown face: a synthetic flat histogram, denied before voice or product
- wrong voice: colombe's face with nelly's voice, denied before product
- unknown voice: colombe's face with a synthetic voice, denied before product

`--save` writes the terminal text, a screenshot of each check, and `scenario_results.csv` to `results/integration/`.

Other checks:

```
python src/system_cli.py --scenario authorized --member nelly --customer-id A178
python src/system_cli.py --scenario wrong-voice --member sahra --voice-member tifare
python src/system_cli.py
```

The last command prompts for an image path and a voice path. Press Enter on both to replay colombe's enrolled feature rows.

### Live image and audio files

Raw face photos and voice recordings are not in this repository. When the private files are available:

```
python src/system_cli.py --image images/raw/colombe_neutral.jpeg --audio audio/raw/colombe_yes_approve.wav --customer-id A178
```

The voice file is always checked as the person named by the face gate. A live unauthorized attempt uses a generated noise image and a generated noise recording:

```
python src/system_cli.py --scenario live-unauthorized --save
```

That file check needs `opencv-python`. The audio file also needs `librosa` (`pip install librosa`). If `ffmpeg` is on PATH, the audio check uses `src/voice_verification.py` directly.

### Demo recording, first cut

Record the terminal while running `python src/system_cli.py --scenario all`. Say, in order:

1. The face gate names the enrolled person and opens the product step.
2. The voice gate accepts only that same person.
3. The category appears only after both passes. Mention that customer 178 is scored from the social profile.
4. Replay the three denials and point out that no category is printed.

Then run one live photo and one live recording if the private files are on the machine, plus `--scenario live-unauthorized`.

Saved model files used by this app:

- `models/face_model.joblib`
- `models/voice_model.pkl`
- `Formative2_Tifare/models/product_recommender.joblib`

## Image processing and face recognition

This part collects face photos, prepares them, and trains a model that recognises each team member.

### Folders

- `src/image_processing.py` - prepares the photos and saves the features
- `src/face_model.py` - trains and tests the face model
- `notebooks/image_processing.ipynb` - runs the image steps and shows the results
- `data/image_features.csv` - the image features
- `models/face_model.joblib` - the saved face model
- `results/` - the face model results

The `images/` folder is not on GitHub. It holds our face photos, so we keep it private.

### Photos

Each member has 3 photos: neutral, smiling and surprised. Members are colombe, nelly, sahra and tifare, so there are 12 photos. Each file is named `name_expression.jpeg`, for example `sahra_smiling.jpeg`. They go in `images/raw/`.

### What the code does

1. Resizes every photo to 224 x 224 and saves it as a .jpg.
2. Makes 3 changed copies of each photo: rotated, flipped and grayscale.
3. Takes features from every photo: a 32-bin brightness histogram, the mean brightness and the standard deviation.
4. Saves all of this in `data/image_features.csv`. There are 48 rows, which is 12 photos with 4 versions each (the original and 3 copies).

The `source_image` column shows which original photo a row came from. When splitting data for training and testing, rows with the same `source_image` must stay together.

### Face model

We used a Random Forest model. To test it, we left out one expression at a time (for example, all the surprised photos), trained on the other two, and checked the left-out photos. Only original photos were used for testing.

To test stranger rejection, we left one member out of training and treated her as a stranger. The model rejects a face when its confidence is below the threshold.

### Results

| Measure | Value |
|---|---|
| Accuracy | 1.00 |
| F1 score | 1.00 |
| Loss | 0.184 |
| Threshold | 0.7 |
| False reject rate | 0.083 (1 of 12 known photos) |
| False accept rate | 0.0 (0 of 12 stranger photos) |

We chose 0.7 because it let in no strangers and refused a known member only once. At 0.8, known members were refused 33% of the time. The full table is in `results/face_threshold_results.csv`.

### Limits

- There are only 12 photos, so the results can change a lot with one wrong prediction.
- The features only measure brightness. Each member has a different background and lighting, so the model may be partly recognising the background and not only the face.
- The stranger test uses a team member, not a real outsider.

### How to run

Install the libraries:

```
pip install -r requirements.txt
```

Prepare the images and features:

```
python src\image_processing.py
```

Train and test the face model:

```
python src\face_model.py
```

To check one photo:

```
python -c "import sys; sys.path.insert(0,'src'); import face_model; print(face_model.identify('images/raw/sahra_neutral.jpeg'))"
```

It prints the member's name and a confidence score. If the face is rejected, the name is `None`.