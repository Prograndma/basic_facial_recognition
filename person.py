import os
from imutils import paths


class Person:

    def __init__(self, name):
        self.name = name
        self.image_vectors = []
        self.image_paths = []

    def add_image_vector(self, image_vector, name):
        self.image_vectors.append(image_vector)
        self.image_paths.append(name)

    def rename_person(self, name):
        dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'

        if self.name in os.listdir(f"{dir_path}data_set") and self.name in os.listdir(f"{dir_path}processed_data"):
            os.rename(f"{dir_path}data_set/{self.name}", f"{dir_path}data_set/{name}")
            os.rename(f"{dir_path}processed_data/{self.name}", f"{dir_path}processed_data/{name}")
            self.name = name
            image_paths = list(paths.list_images(f"{dir_path}data_set/{self.name}"))
            image_paths.sort()
            self.image_paths = []
            for image_path in image_paths:
                self.image_paths.append(image_path)
        else:
            raise Exception("problem", f"Could not find {dir_path}data_set/{self.name} or {dir_path}processed_data/{self.name}\n"
                            f"Name not changed")

    def __str__(self):
        return self.name + "\'s person object"

    def __repr__(self):
        return self.__str__()
