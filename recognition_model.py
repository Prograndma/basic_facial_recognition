from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
import pickle
import os
import time

dir_path = os.path.dirname(os.path.realpath(__file__)) + '/face_recognition_models/'


class RecognitionModel:
    def __init__(self, name, train, accuracy=90):
        self.name = name
        if train:
            self.recognizer, self.encoder = self.train_model()
        else:
            self.recognizer, self.encoder = self.load_recognizer()
        self.accuracy = accuracy

    def train_model(self):
        start = time.time()
        embed_path = f"{dir_path}{self.name}/data/embeddings.pickle"
        recognize_path = f"{dir_path}{self.name}/recognizer.pickle"
        label_path = f"{dir_path}{self.name}/labels.pickle"

        print("[INFO] loading face embeddings...")
        data = pickle.loads(open(embed_path, "rb").read())

        print("[INFO] encoding labels...")
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(data["names"])

        print("[INFO] training model...")
        recognizer = SVC(C=1.0, kernel="rbf", probability=True)
        recognizer.fit(data["embeddings"], labels)

        # write the actual face recognition model to disk
        f = open(recognize_path, "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        file = open(label_path, "wb")
        file.write(pickle.dumps(label_encoder))
        file.close()
        print(f"Model trained for {self.name}\nTime elapsed:{time.time() - start}")

        return recognizer, label_encoder

    def load_recognizer(self):
        recognize_path = f"{dir_path}{self.name}/recognizer.pickle"
        label_path = f"{dir_path}{self.name}/labels.pickle"
        return pickle.loads(open(recognize_path, "rb").read()), pickle.loads(open(label_path, "rb").read())

    def recognize_face(self, vec):

        predictions = self.recognizer.predict_proba(vec)[0]

        index = np.argmax(predictions)
        probability = predictions[index] * 100

        name = self.encoder.classes_[index]

        if probability < self.accuracy:
            probability = (((probability - self.accuracy) * -1) * (10/(self.accuracy / 10)))

            return probability, -1

        probability = (probability - self.accuracy) * 10
        return probability, name

    def closest_face(self, vec):
        predictions = self.recognizer.predict_proba(vec)[0]

        index = np.argmax(predictions)
        probability = predictions[index] * 100

        name = self.encoder.classes_[index]

        return probability, name
