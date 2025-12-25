import os
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# ==============================================================================
# Helper Function: Display Images
# ==============================================================================
def show_img(path, title="Image"):
    """
    Reads and displays an image using Matplotlib.
    DeepFace works with file paths, but visualization helps us understand the result.
    """
    if os.path.exists(path):
        img = cv2.imread(path)
        # Convert BGR (OpenCV default) to RGB (Matplotlib default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        print(f"Error: Image not found at {path}")

# ==============================================================================
# 1. FACE VERIFICATION (One-to-One)
# ==============================================================================
# Goal: Check if two images belong to the same person.
# Scenario: Unlocking a phone with your face.

print("\n--- 1. Running Face Verification ---")

img1_path = "img1.jpg"  # Replace with your image path
img2_path = "img2.jpg"  # Replace with another image of the SAME person

# We use the 'VGG-Face' model by default. It is balanced between speed and accuracy.
# 'detector_backend' finds the face in the image. 'opencv' is fast; 'retinaface' is accurate.
result = DeepFace.verify(
    img1_path=img1_path,
    img2_path=img2_path,
    model_name="VGG-Face",
    detector_backend="opencv"
)

# The result is a dictionary containing 'verified' (True/False) and 'distance'.
print(f"Is it the same person? {result['verified']}")
print(f"Similarity Distance: {result['distance']}")
# Note: Lower distance means higher similarity.

# ==============================================================================
# 2. FACE ATTRIBUTE ANALYSIS
# ==============================================================================
# Goal: Extract Demographics (Age, Gender, Emotion, Race).
# Scenario: Targeted advertising or crowd analytics.

print("\n--- 2. Running Face Analysis ---")

# We can analyze specific attributes using the 'actions' parameter.
analysis = DeepFace.analyze(
    img_path=img1_path,
    actions=['age', 'gender', 'emotion', 'race'],
    detector_backend="opencv",
    enforce_detection=False # Set to False to prevent crash if no face is detected
)

# DeepFace returns a list (in case there are multiple faces). We take the first one.
first_face = analysis[0]

print(f"Estimated Age: {first_face['age']}")
print(f"Dominant Gender: {first_face['dominant_gender']}")
print(f"Dominant Emotion: {first_face['dominant_emotion']}")
print(f"Dominant Race: {first_face['dominant_race']}")

# ==============================================================================
# 3. FACE RECOGNITION (Find / One-to-Many)
# ==============================================================================
# Goal: Find a specific face inside a database (folder of many images).
# Scenario: Police searching for a suspect in a database or Employee Attendance.

print("\n--- 3. Running Face Recognition (Find) ---")

target_img = "img1.jpg"
database_path = "my_db" # Create a folder named 'my_db' and put some face images in it

if os.path.exists(database_path):
    # This function returns a List of Pandas DataFrames.
    # It scans the folder, creates embeddings for all images, and matches them.
    dfs = DeepFace.find(
        img_path=target_img,
        db_path=database_path,
        model_name="VGG-Face",
        detector_backend="opencv"
    )

    if len(dfs) > 0 and not dfs[0].empty:
        print(f"Found matches in database:")
        # The dataframe contains paths to the images that matched the target
        print(dfs[0].head()) 
    else:
        print("No matches found in the database.")
else:
    print(f"Skipping Step 3: Database folder '{database_path}' not found.")

# ==============================================================================
# 4. FACE REPRESENTATION (Embeddings)
# ==============================================================================
# Goal: Convert a face into a vector (list of numbers).
# Scenario: Storing faces in a database securely without keeping the actual image.

print("\n--- 4. Generating Face Embeddings ---")

embedding_objs = DeepFace.represent(
    img_path=img1_path,
    model_name="VGG-Face",
    detector_backend="opencv"
)

embedding = embedding_objs[0]["embedding"]
print(f"Vector size: {len(embedding)}") 
print(f"First 5 values of vector: {embedding[:5]}")
# This vector is what the AI actually "sees". You can compare these vectors using
# Cosine Similarity to do your own custom matching.