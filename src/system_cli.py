"""Command-line flow that chains face, voice, and product checks.

A product category is shown only after the face model recognizes an enrolled
person and the voice model confirms that the same person spoke.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
TIFARE_SRC = ROOT / "Formative2_Tifare" / "src"
FACE_MODEL = ROOT / "models" / "face_model.joblib"
VOICE_MODEL = ROOT / "models" / "voice_model.pkl"
PRODUCT_MODEL = ROOT / "Formative2_Tifare" / "models" / "product_recommender.joblib"
IMAGE_FEATURES = ROOT / "data" / "image_features.csv"
AUDIO_FEATURES = ROOT / "data" / "audio_features.csv"
SOCIAL_PROFILES = ROOT / "Formative2_Tifare" / "data" / "raw" / "customer_social_profiles.csv"
RESULTS_DIR = ROOT / "results" / "integration"
ENROLLED = ("colombe", "nelly", "sahra", "tifare")
_CACHE = {}


@dataclass
class CheckResult:
    title: str
    allowed: bool
    reason: str
    transcript: str
    face_name: str | None = None
    face_confidence: float | None = None
    nearest_face: str | None = None
    voice: dict | None = None
    prediction: dict | None = None
    prediction_called: bool = False


def _load_model(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


def load_face_bundle():
    if "face" not in _CACHE:
        _CACHE["face"] = _load_model(FACE_MODEL)
    return _CACHE["face"]


def load_voice_bundle():
    if "voice" not in _CACHE:
        _CACHE["voice"] = _load_model(VOICE_MODEL)
    return _CACHE["voice"]


def load_product_bundle():
    if "product" not in _CACHE:
        _CACHE["product"] = _load_model(PRODUCT_MODEL)
    return _CACHE["product"]


def display_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def model_name(model):
    if hasattr(model, "steps"):
        return type(model.steps[-1][1]).__name__
    return type(model).__name__


def gate_summary():
    face = load_face_bundle()
    voice = load_voice_bundle()
    product = load_product_bundle()
    return [
        f"Face gate: {model_name(face['model'])}, accept at or above {float(face['threshold']):.3f}",
        f"Voice gate: {model_name(voice['model'])}, accept at or above {float(voice['threshold']):.3f}",
        f"Product model: {model_name(product['model'])}",
        "Rule: show a product only when the face and the voice are the same enrolled person.",
    ]


def score_face(features):
    """Score one image feature row with the saved face model."""
    bundle = load_face_bundle()
    frame = pd.DataFrame([features])[bundle["columns"]]
    probs = bundle["model"].predict_proba(frame)[0]
    best = int(np.argmax(probs))
    nearest = str(bundle["model"].classes_[best])
    confidence = float(probs[best])
    threshold = float(bundle["threshold"])
    accepted = nearest if confidence >= threshold else None
    return accepted, confidence, nearest, threshold


def score_voice(features, claimed):
    """Score one audio feature row against the person the face gate named."""
    bundle = load_voice_bundle()
    frame = pd.DataFrame([features])[bundle["features"]]
    proba = bundle["model"].predict_proba(frame)[0]
    classes = [str(label) for label in bundle["model"].classes_]
    claimed = str(claimed).lower()
    score = float(proba[classes.index(claimed)]) if claimed in classes else 0.0
    predicted = classes[int(np.argmax(proba))]
    threshold = float(bundle["threshold"])
    return {
        "approved": score >= threshold,
        "score": round(score, 3),
        "predicted": predicted,
        "claimed": claimed,
        "threshold": threshold,
    }


def image_table():
    if "images" not in _CACHE:
        _CACHE["images"] = pd.read_csv(IMAGE_FEATURES)
    return _CACHE["images"]


def audio_table():
    if "audio" not in _CACHE:
        _CACHE["audio"] = pd.read_csv(AUDIO_FEATURES)
    return _CACHE["audio"]


def _one_row(table, kind, **filters):
    found = table
    for column, value in filters.items():
        found = found[found[column].astype(str).str.lower() == str(value).lower()]
    if found.empty:
        raise ValueError(f"No {kind} row for {filters}")
    return found.iloc[0]


def enrolled_face(member, expression="neutral", augmentation="original"):
    bundle = load_face_bundle()
    row = _one_row(
        image_table(),
        "image",
        member=member,
        expression=expression,
        augmentation=augmentation,
    )
    features = {column: row[column] for column in bundle["columns"]}
    return score_face(features)


def enrolled_voice(speaker, phrase="yes_approve", augmentation="none", claimed=None):
    bundle = load_voice_bundle()
    row = _one_row(
        audio_table(),
        "audio",
        speaker=speaker,
        phrase=phrase,
        augmentation=augmentation,
    )
    features = {column: row[column] for column in bundle["features"]}
    return score_voice(features, claimed or speaker)


def unauthorized_face_features():
    columns = load_face_bundle()["columns"]
    features = {}
    for column in columns:
        if str(column).startswith("hist_"):
            features[column] = 1 / 32
        elif column == "mean_intensity":
            features[column] = 128.0
        elif column == "std_intensity":
            features[column] = 40.0
        else:
            features[column] = 0.0
    return features


def unauthorized_voice_features():
    return {column: 0.0 for column in load_voice_bundle()["features"]}


def social_table():
    """Clean social profiles with the same id and text rules as the product pipeline."""
    if "social" in _CACHE:
        return _CACHE["social"]
    if str(TIFARE_SRC) not in sys.path:
        sys.path.insert(0, str(TIFARE_SRC))
    from pipeline import normalize_id

    social = pd.read_csv(SOCIAL_PROFILES).drop_duplicates().copy()
    social["customer_id"] = social.customer_id_new.map(normalize_id)
    for column in ("engagement_score", "purchase_interest_score"):
        social[column] = pd.to_numeric(social[column], errors="coerce")
    social["social_media_platform"] = social.social_media_platform.astype("string").str.strip().str.title()
    social["review_sentiment"] = social.review_sentiment.astype("string").str.strip().str.title()
    _CACHE["social"] = social
    return social


def customer_features(customer_id):
    """Build the pre-purchase social features for one visitor.

    Customer A178 has a social profile and no row in the transaction model
    table. The recommendation uses that profile, not a purchase that already
    happened.
    """
    if str(TIFARE_SRC) not in sys.path:
        sys.path.insert(0, str(TIFARE_SRC))
    from pipeline import FEATURES, most_common, normalize_id

    canonical = normalize_id(customer_id)
    if pd.isna(canonical):
        raise ValueError(f"Customer id {customer_id!r} is not a valid id")
    subset = social_table()
    subset = subset[subset.customer_id == canonical]
    if subset.empty:
        raise ValueError(f"Customer {customer_id} was not found in the social profiles")
    row = {
        "engagement_mean": float(subset.engagement_score.mean()),
        "interest_mean": float(subset.purchase_interest_score.mean()),
        "social_activity_count": float(len(subset)),
        "distinct_platforms": float(subset.social_media_platform.nunique(dropna=True)),
        "primary_platform": most_common(subset.social_media_platform),
        "primary_sentiment": most_common(subset.review_sentiment),
    }
    return pd.DataFrame([row])[FEATURES], str(canonical)


def predict_product(customer_id):
    features, canonical = customer_features(customer_id)
    bundle = load_product_bundle()
    model = bundle["model"]
    row = features[bundle["features"]]
    probabilities = model.predict_proba(row)[0]
    classes = [str(label) for label in model.classes_]
    order = np.argsort(probabilities)[::-1]
    return {
        "customer_id": canonical,
        "category": str(model.predict(row)[0]),
        "probabilities": [(classes[int(index)], float(probabilities[int(index)])) for index in order],
    }


def decide(face_name, voice):
    """Return whether this face and voice may open the product model."""
    if face_name is None:
        return False, "The face was not recognized, so the product model stays closed."
    if voice is None:
        return False, "A matching voice sample is required before a product can be shown."
    claimed = str(voice.get("claimed", "")).lower()
    predicted = str(voice.get("predicted", "")).lower()
    face_name = str(face_name).lower()
    if claimed != face_name:
        return False, f"The voice was checked as {claimed or 'nobody'}, not as {face_name}."
    if predicted != face_name:
        return False, (
            f"The voice does not match {face_name}. "
            f"Closest enrolled speaker: {predicted}. "
            f"Score for {face_name}: {voice.get('score')}."
        )
    if not voice.get("approved", False):
        return False, (
            f"The voice score for {face_name} is {voice.get('score')} "
            f"which is below {voice.get('threshold')}."
        )
    return True, f"Face and voice both match {face_name}."


def run_check(title, face, voice, customer_id, face_input, voice_input, predict=predict_product):
    accepted, confidence, nearest, threshold = face
    allowed, reason = decide(accepted, voice)
    prediction = None
    called = False
    if allowed:
        called = True
        try:
            prediction = predict(customer_id)
        except ValueError as error:
            allowed = False
            reason = str(error)
    transcript = render(
        title,
        face_input,
        voice_input,
        accepted,
        confidence,
        nearest,
        threshold,
        voice,
        customer_id,
        prediction,
        allowed,
        reason,
    )
    return CheckResult(
        title=title,
        allowed=allowed,
        reason=reason,
        transcript=transcript,
        face_name=accepted,
        face_confidence=confidence,
        nearest_face=nearest,
        voice=voice,
        prediction=prediction,
        prediction_called=called,
    )


def render(
    title,
    face_input,
    voice_input,
    accepted,
    confidence,
    nearest,
    threshold,
    voice,
    customer_id,
    prediction,
    allowed,
    reason,
):
    lines = gate_summary()
    lines.append("")
    lines.append(title)
    lines.append("-" * len(title))
    lines.append("")
    lines.append("Step 1 of 3. Face recognition")
    lines.append(f"Input: {face_input}")
    lines.append(f"Nearest enrolled person: {nearest}")
    lines.append(f"Confidence: {confidence:.3f}")
    lines.append(f"Threshold: {threshold:.3f}")
    if accepted is None:
        lines.append("Decision: FAIL. This face is not an enrolled person.")
        lines.append("")
        lines.append("ACCESS DENIED")
        lines.append(reason)
        lines.append("Voice verification and product prediction were not run.")
        return "\n".join(lines)

    lines.append(f"Decision: PASS. {accepted} may request a product prediction.")
    lines.append("")
    lines.append("Step 2 of 3. Voice verification")
    if voice is None:
        lines.append("Input: none")
        lines.append("Decision: FAIL. No voice sample was provided.")
        lines.append("")
        lines.append("ACCESS DENIED")
        lines.append(reason)
        lines.append("Product prediction was not run.")
        return "\n".join(lines)

    lines.append(f"Input: {voice_input}")
    lines.append(f"Voice is checked as: {voice['claimed']}")
    lines.append(f"Predicted speaker: {voice['predicted']}")
    lines.append(f"Score for {voice['claimed']}: {voice['score']:.3f}")
    lines.append(f"Threshold: {voice['threshold']:.3f}")
    voice_matches = (
        str(voice["claimed"]).lower() == accepted
        and str(voice["predicted"]).lower() == accepted
        and bool(voice["approved"])
    )
    if not voice_matches:
        lines.append("Decision: FAIL. The voice does not match the recognized face.")
        lines.append("")
        lines.append("ACCESS DENIED")
        lines.append(reason)
        lines.append("Product prediction was not run.")
        return "\n".join(lines)

    lines.append("Decision: PASS. The voice is the same enrolled person as the face.")
    lines.append("")
    lines.append("Step 3 of 3. Product recommendation")
    lines.append(f"Customer: {customer_id}")
    if prediction is None:
        lines.append("Decision: FAIL. No category was predicted.")
        lines.append("")
        lines.append("ACCESS DENIED")
        lines.append(reason)
        return "\n".join(lines)

    lines.append(f"Predicted category: {prediction['category']}")
    lines.append("Class probabilities:")
    for label, probability in prediction["probabilities"]:
        lines.append(f"  {label}: {probability:.1%}")
    lines.append("The category comes from the social profile only. Purchase amount,")
    lines.append("rating, and the past product label are not model inputs.")
    lines.append("")
    lines.append("ACCESS GRANTED")
    lines.append(reason)
    return "\n".join(lines)


def scenario_authorized(member="colombe", expression="neutral", phrase="yes_approve", customer_id="178", predict=predict_product):
    member = member.lower()
    return run_check(
        f"Authorized transaction for {member}",
        enrolled_face(member, expression),
        enrolled_voice(member, phrase, claimed=member),
        customer_id,
        f"enrolled original photo features for {member} ({expression})",
        f"enrolled original voice features for {member} ({phrase})",
        predict=predict,
    )


def scenario_unknown_face(customer_id="178", predict=predict_product):
    scored = score_face(unauthorized_face_features())
    return run_check(
        "Unauthorized face",
        scored,
        None,
        customer_id,
        "synthetic face with a flat brightness histogram",
        "not checked",
        predict=predict,
    )


def other_member(member):
    member = member.lower()
    for person in ENROLLED:
        if person != member:
            return person
    raise ValueError(f"{member} is not an enrolled person")


def scenario_wrong_voice(member="colombe", voice_member=None, expression="neutral", phrase="yes_approve", customer_id="178", predict=predict_product):
    member = member.lower()
    voice_member = (voice_member or other_member(member)).lower()
    if voice_member == member:
        voice_member = other_member(member)
    return run_check(
        f"Known face ({member}) with another member's voice ({voice_member})",
        enrolled_face(member, expression),
        enrolled_voice(voice_member, phrase, claimed=member),
        customer_id,
        f"enrolled original photo features for {member} ({expression})",
        f"enrolled original voice features for {voice_member} ({phrase})",
        predict=predict,
    )


def scenario_unknown_voice(member="colombe", expression="neutral", customer_id="178", predict=predict_product):
    member = member.lower()
    return run_check(
        f"Known face ({member}) with an unknown voice",
        enrolled_face(member, expression),
        score_voice(unauthorized_voice_features(), member),
        customer_id,
        f"enrolled original photo features for {member} ({expression})",
        "synthetic unknown voice (zero MFCC, roll-off, and energy)",
        predict=predict,
    )


def all_scenarios(member="colombe", customer_id="178", voice_member=None, predict=predict_product):
    return [
        scenario_authorized(member=member, customer_id=customer_id, predict=predict),
        scenario_unknown_face(customer_id=customer_id, predict=predict),
        scenario_wrong_voice(member=member, voice_member=voice_member, customer_id=customer_id, predict=predict),
        scenario_unknown_voice(member=member, customer_id=customer_id, predict=predict),
    ]


def identify_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import cv2
    import image_processing

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.resize(image, image_processing.SIZE)
    return score_face(image_processing.extract_features(image))


def verify_audio(path, claimed):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import voice_verification

    if shutil.which("ffmpeg"):
        return voice_verification.verify_voice(str(path), claimed, model_path=str(VOICE_MODEL))
    import librosa

    audio, sample_rate = librosa.load(str(path), sr=16000, mono=True)
    return score_voice(voice_verification._extract(audio, sample_rate), claimed)


def run_files(image_path, audio_path, customer_id, predict=predict_product):
    face = identify_image(image_path)
    accepted = face[0]
    voice = None
    voice_input = "not checked, because the face was rejected"
    if accepted is not None:
        voice = verify_audio(audio_path, accepted)
        voice_input = display_path(audio_path)
    return run_check(
        "Live face and voice check",
        face,
        voice,
        customer_id,
        display_path(image_path),
        voice_input,
        predict=predict,
    )


def write_unauthorized_samples(directory=RESULTS_DIR):
    """Write a synthetic face image and a synthetic voice file. Neither is a person."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / "unauthorized_face.jpg"
    audio_path = directory / "unauthorized_voice.wav"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import cv2

    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    samples = (rng.normal(0, 0.01, 16000) * 32767).astype(np.int16)
    import wave

    with wave.open(str(audio_path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(samples.tobytes())
    return image_path, audio_path


def scenario_live_unauthorized(member="colombe", customer_id="178", predict=predict_product):
    image_path, audio_path = write_unauthorized_samples()
    face_only = run_check(
        "Unauthorized image file",
        identify_image(image_path),
        None,
        customer_id,
        display_path(image_path),
        "not checked",
        predict=predict,
    )
    known_face = enrolled_face(member)
    try:
        voice = verify_audio(audio_path, member)
    except ImportError as error:
        raise ImportError(
            f"{error}. Install librosa, or add ffmpeg to PATH, to score the unauthorized audio file."
        ) from error
    voice_case = run_check(
        f"Enrolled face ({member}) with an unauthorized audio file",
        known_face,
        voice,
        customer_id,
        f"enrolled original photo features for {member} (neutral)",
        display_path(audio_path),
        predict=predict,
    )
    return [face_only, voice_case]


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined = []
    rows = []
    for result in results:
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in result.title).strip("_")
        text_path = RESULTS_DIR / f"{slug}.txt"
        text_path.write_text(result.transcript + "\n", encoding="utf-8")
        save_png(result.transcript, RESULTS_DIR / f"{slug}.png")
        combined.append(result.transcript)
        voice = result.voice or {}
        prediction = result.prediction or {}
        rows.append({
            "scenario": result.title,
            "allowed": result.allowed,
            "reason": result.reason,
            "face": result.face_name,
            "face_confidence": result.face_confidence,
            "voice_predicted": voice.get("predicted"),
            "voice_score": voice.get("score"),
            "voice_approved": voice.get("approved"),
            "customer_id": None if prediction is None else prediction.get("customer_id"),
            "predicted_category": None if prediction is None else prediction.get("category"),
        })
    (RESULTS_DIR / "demo_transcript.txt").write_text("\n\n".join(combined) + "\n", encoding="utf-8")
    save_png("\n\n".join(combined), RESULTS_DIR / "demo_transcript.png")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "scenario_results.csv", index=False)
    return RESULTS_DIR


def save_png(text, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = text.splitlines() or [""]
    width = max(len(line) for line in lines)
    figure = plt.figure(figsize=(min(14, max(8, width * 0.078)), max(2.2, len(lines) * 0.16 + 0.35)), dpi=140)
    figure.patch.set_facecolor("white")
    axis = figure.add_axes((0.02, 0.02, 0.96, 0.96))
    axis.axis("off")
    axis.text(0, 1, text, va="top", ha="left", family="monospace", fontsize=8, color="black")
    figure.savefig(path)
    plt.close(figure)


def print_results(results):
    print("\n\n".join(result.transcript for result in results))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the face gate, then the voice gate, then the product model."
    )
    parser.add_argument(
        "--scenario",
        choices=("authorized", "unknown-face", "wrong-voice", "unknown-voice", "live-unauthorized", "all"),
        help="Scripted check. 'all' runs the authorized path and three denied paths.",
    )
    parser.add_argument("--member", default="colombe", help="Enrolled person for the face. Default: colombe.")
    parser.add_argument("--voice-member", help="Speaker used by the wrong-voice scenario.")
    parser.add_argument("--customer-id", default="178", help="Social-profile customer to score. Default: 178.")
    parser.add_argument("--image", help="Face image for a live check.")
    parser.add_argument("--audio", help="Voice recording for a live check. Checked as the recognized face.")
    parser.add_argument("--save", action="store_true", help="Write transcripts and screenshots to results/integration.")
    return parser


def selected_results(args):
    if args.image or args.audio:
        if not args.image or not args.audio:
            raise ValueError("A live check needs both --image and --audio.")
        return [run_files(args.image, args.audio, args.customer_id)]
    if args.scenario == "authorized":
        return [scenario_authorized(member=args.member, customer_id=args.customer_id)]
    if args.scenario == "unknown-face":
        return [scenario_unknown_face(customer_id=args.customer_id)]
    if args.scenario == "wrong-voice":
        return [scenario_wrong_voice(member=args.member, voice_member=args.voice_member, customer_id=args.customer_id)]
    if args.scenario == "unknown-voice":
        return [scenario_unknown_voice(member=args.member, customer_id=args.customer_id)]
    if args.scenario == "live-unauthorized":
        return scenario_live_unauthorized(member=args.member, customer_id=args.customer_id)
    if args.scenario == "all" or args.save:
        return all_scenarios(member=args.member, customer_id=args.customer_id, voice_member=args.voice_member)
    return None


def interactive(customer_default="178"):
    print("Multimodal identity check")
    print("Press Enter to replay colombe's enrolled samples, or type file paths.")
    image = input("Face image: ").strip()
    audio = input("Voice sample: ").strip()
    customer = input(f"Customer id [{customer_default}]: ").strip() or customer_default
    if image or audio:
        if not image or not audio:
            print("A live check needs both a face image and a voice sample.")
            return 1
        results = [run_files(image, audio, customer)]
    else:
        results = [scenario_authorized(customer_id=customer)]
    print_results(results)
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        results = selected_results(args)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as error:
        print(f"Error: {error}")
        return 1
    if results is None:
        return interactive(args.customer_id)
    print_results(results)
    if args.save:
        folder = save_results(results)
        print(f"\nSaved transcripts and screenshots to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
