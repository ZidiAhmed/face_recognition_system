import cv2
import face_recognition
import os

def load_known_faces(folder_path):
    known_faces = []
    known_names = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            face_image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(face_image)[0]

            known_faces.append(face_encoding)
            known_names.append(os.path.splitext(filename)[0])

    return known_faces, known_names

def recognize_faces(image_path, known_faces, known_names):
    unknown_image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        print(f"Person recognized: {name}")

def main():
    stuff_image_folder = "stuff_image"
    known_faces, known_names = load_known_faces(stuff_image_folder)

    test_image_path = "path/to/your/test/image.jpg"
    recognize_faces(test_image_path, known_faces, known_names)

if __name__ == "__main__":
    main()
