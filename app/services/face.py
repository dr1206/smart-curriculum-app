try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False
import numpy as np
from typing import Optional
import io

def compute_face_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Compute face embedding from image bytes using face_recognition library.
    Returns None if no face detected or library not available.
    """
    if not HAS_FACE_RECOGNITION:
        print("face_recognition library not available. Using random embedding for demo.")
        return np.random.rand(128)
    try:
        # Load image from bytes
        image = face_recognition.load_image_file(io.BytesIO(image_bytes))

        # Find face encodings
        face_encodings = face_recognition.face_encodings(image)

        if not face_encodings:
            return None

        # Return the first face encoding
        return face_encodings[0]
    except Exception as e:
        print(f"Error computing face embedding: {e}")
        return None

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    """
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    # Normalize
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)

    return np.dot(emb1_norm, emb2_norm)
