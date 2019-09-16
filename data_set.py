import os
import time
from imutils import paths
from face_aligner import Aligner
import cv2 as cv
from person import Person
import numpy as np
import datetime
from pathlib import Path
from plotting import Plotter
import pickle
from recognition_model import RecognitionModel

dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'

embedding_tool = cv.dnn.readNetFromTorch(f"{dir_path}nn4.small2.v1.t7")


class DataSet:

    def __init__(self, name, save_flag=False, directory='data_sets/'):
        self.name = name
        self.save_flag = save_flag
        self.person_list = []
        self.directory = directory
        self.recognition_model = None

    def is_in_data_set(self, image):
        face_blob = cv.dnn.blobFromImage(image, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedding_tool.setInput(face_blob)
        vec = embedding_tool.forward()
        confidence, index = self.recognition_model.recognize_face(vec)
        if index == -1:
            name = "Unknown"
            is_match = False
        else:
            name = index
            is_match = True

        return is_match, confidence, name

    def is_vector_in_data_set(self, vec):
        confidence, index = self.recognition_model.recognize_face(vec)
        if index == -1:
            name = "Unknown"
            is_match = False
        else:
            name = index
            is_match = True

        return is_match, confidence, name

    def closest_person_in_data_set(self, image):
        face_blob = cv.dnn.blobFromImage(image, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedding_tool.setInput(face_blob)
        vec = embedding_tool.forward()
        return self.recognition_model.closest_face(vec)

    def add_person_to_list(self, person):
        if isinstance(self.person_list, list):
            self.person_list.append(person)
        else:
            self.person_list = np.append(self.person_list, person)

    def save_embeddings(self):
        names = []
        embeddings = []
        for person in self.person_list:
            for embedded_face in person.image_vectors:
                embedded_face = embedded_face.flatten()
                names.append(person.name)
                embeddings.append(embedded_face)
        data = {"embeddings": embeddings, "names": names}
        file = open(f"{dir_path}face_recognition_models/{self.name}/data/embeddings.pickle", "wb")
        file.write(pickle.dumps(data))
        file.close()
        print('embeddings have been saved')

    def has_person_named(self, name):
        for person in self.person_list:
            if person.name == name:
                return True
        return False

    def get_person_named(self, name):
        for person in self.person_list:
            if person.name == name:
                return person
        raise AttributeError

    def call_on_each_person(self, method_to_call, **kwargs):
        for person in self.person_list:
            method_to_call(person, **kwargs)

    def set_up(self):
        start = time.time()
        image_paths = list(paths.list_images(f"{dir_path}{self.directory}/{self.name}"))
        aligner = Aligner()

        image_paths.sort()
        total_paths = len(image_paths)

        for i, image_path in enumerate(image_paths):
            name, image_name, image, created_person = self._get_name_image_from_path(image_path, i, total_paths, aligner)

            if created_person:
                yield ('debug', f"Created new person object {name}")

            if image is None:
                yield ('warn', f"could not load image from {image_path}")
                continue

            if not self._add_embedding_to_person(image, image_path, name):
                yield ('error', f"could not embed image from {image_path}")

            if self.save_flag:

                if not cv.imwrite(f"{dir_path}processed_data/{self.name}/{name}/{image_name}_{i}.png", image):
                    yield ('warn', f"could not save aligned image to {dir_path}processed_data/{self.name}/{name}/{i}.png")
                    print("could not save image")
                    print(f"path:{dir_path}processed_data/{self.name}/{name}/{i}.png")

        print(f"\nEmbedded all of {self.person_list[-1].name}'s pictures\n")

        print(f"total time elapsed: {time.time() - start}")
        for i, person in enumerate(self.person_list):
            print(f"{i}: {person}")
        self.save_embeddings()
        self.recognition_model = RecognitionModel(self.name, train=True)
        return

    def _get_name_image_from_path(self, path, position, length, aligner):
        print(f"{position + 1} of {length} images embedded")
        # The person's name should be the name of the folder they're in. (path[-2])
        name = path.split(os.path.sep)[-2]
        pic_name = path.split(os.path.sep)[-1]
        created_person = False

        if len(self.person_list) == 0:
            created_person = True
            self.add_person_to_list(Person(name))
        elif not self.has_person_named(name):
            self.add_person_to_list(Person(name))
            created_person = True
            print(f"\nembedded all of {self.person_list[-2].name}'s pictures\n")
        image = cv.imread(path)
        if image is None:
            print(f"ERROR: Could not load image from path:{path}")
        image = aligner.align_image(image)

        return name, pic_name, image, created_person

    def _add_embedding_to_person(self, image, path, name=None):
        face_blob = cv.dnn.blobFromImage(image, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedding_tool.setInput(face_blob)

        vec = embedding_tool.forward()
        if vec is None:
            return False

        if name is None:
            self.person_list[-1].add_image_vector(vec, path)
        else:
            self.get_person_named(name).add_image_vector(vec, path)
        return True

    # def add_images_to_person(self, name, pictures, vectors, aligned_images):
    #     person = self.get_person_named(name)
    #     print(f"Embedding new images and adding them to {person.name}")
    #     save_path = f"{dir_path}{self.directory}{self.name}/{person.name}/dynamic_capture_at_"
    #     write_path = f"{dir_path}processed_data/{self.name}/{person.name}/dynamic_image"
    #
    #     for i, picture in enumerate(pictures):
    #         print(f"{i + 1}/{len(pictures)} images embedded")
    #         current_time = datetime.datetime.now()
    #         person.add_image_vector(vectors[i], save_path + str(current_time))
    #
    #         if self.save_flag:
    #             if not cv.imwrite(f"{write_path}{current_time}.png", aligned_images[i]):
    #                 print("could not save aligned image")
    #                 print(f"{write_path}{current_time}.png")
    #                 yield ('warning', f"Could not save aligned image at:\n\t{write_path}{current_time}.png")
    #             if not cv.imwrite(f"{save_path}{current_time}.png", picture):
    #                 print("could not save normal image")
    #                 print(f"{save_path}{current_time}.png")
    #                 yield ('warning', f"Could not save normal image at:\n\t{save_path}{current_time}.png")

    # def plot_faces(self):
    #     print(f"creating TSNE plot of {self.name}'s person_list: length {len(self.person_list)}")
    #     begin = time.time()
    #     plotter = Plotter(self.person_list)
    #     yield ('debug', f"creating a t-sne plot of all people in global person_list")
    #     plotter.plot_embedding()
    #     print(f"plot created, time elapsed: {time.time() - begin}")
    #     yield ('debug', f"plot created, time elapsed: {time.time() - begin}")

    # def create_new_person(self, pictures, vectors, aligned_images):
    #     name = f"Dynamic {datetime.datetime.now()}"
    #     person = Person(name)
    #     print(f"Embedding new images and adding them to {person.name}")
    #     save_path = f"{dir_path}{self.directory}{self.name}/{person.name}"
    #     write_path = f"{dir_path}processed_data/{self.name}/{person.name}"
    #     if self.save_flag:
    #         try:
    #             Path(save_path).mkdir(parents=True)
    #         except FileExistsError:
    #             print("ERROR: Could not make directory")
    #             yield ('error', f"could not make directory at\n\t{save_path}")
    #             return
    #         save_path = f"{save_path}/dynamic_capture_at_"
    #         try:
    #             Path(write_path).mkdir(parents=True)
    #         except FileExistsError:
    #             print("ERROR: Could not make directory")
    #             yield ('error', f"could not make directory at\n\t{write_path}")
    #         write_path = f"{write_path}/dynamic_image"
    #
    #         for i, picture in enumerate(pictures):
    #             print(f"{i + 1}/{len(pictures)} images embedded")
    #             current_time = datetime.datetime.now()
    #             if not cv.imwrite(f"{write_path}_{current_time}.png", aligned_images[i]):
    #                 print("could not save aligned image")
    #                 print(f"{write_path}_{current_time}.png")
    #                 yield ('warning', f"could not save aligned image at\n\t{write_path}_{current_time}.png")
    #             if not cv.imwrite(f"{save_path}{current_time}.png", picture):
    #                 print("could not save normal image")
    #                 print(f"{save_path}{current_time}.png")
    #                 yield ('warning', f"could not save normal image at\n\t{save_path}{current_time}.png")
    #             else:
    #                 person.add_image_vector(vectors[i], f"{save_path}{current_time}.png")
    #     else:
    #         for i, picture in enumerate(pictures):
    #             print(f"{i + 1}/{len(pictures)} images embedded")
    #             current_time = datetime.datetime.now()
    #             person.add_image_vector(vectors[i], save_path + str(current_time))
    #
    #     yield ('debug', f"adding new person {name} to {self.name} data_set object : save_flag={self.save_flag}")
    #     self.add_person_to_list(person)
    #     self.save_embeddings()
    #     self.recognition_model = RecognitionModel(self.name, train=True)
