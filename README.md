# Multimodal Identity and Product Recommendation

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

Each member has 3 photos: neutral, smiling and surprised. Members are colombe, nelly, sahra and tifare, so there are 12 photos. Each file is named `name_expression.jpg`, for example `sahra_smiling.jpg`. They go in `images/raw/`.

### What the code does

1. Resizes every photo to 224 x 224 and saves it as a .jpeg.
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