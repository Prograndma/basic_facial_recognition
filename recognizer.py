import os
import numpy as np
from argparse import ArgumentParser
import logging.handlers
import cv2 as cv
from data_set import DataSet
from face_aligner import Aligner
import imutils
from test_set import TestSet

log = 'log.log'
dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'
proto_path = f"{dir_path}face_detection_model/deploy.prototxt"
model_path = f"{dir_path}face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"

detector = cv.dnn.readNetFromCaffe(proto_path, model_path)
embedding_tool = cv.dnn.readNetFromTorch(f"{dir_path}nn4.small2.v1.t7")


def run_test_helper(person, test_data_set=None):
    print(f"\n\nFOR PERSON:{person.name} -- IN SET: {test_data_set.has_person_named(person.name)}")
    num_vectors = len(person.image_vectors)
    num_imposters = 0
    num_betrayals = 0
    num_confusions = 0
    for vector in person.image_vectors:
        is_match, confidence, name = test_data_set.is_vector_in_data_set(vector)
        if is_match:
            if person.name != name:
                if test_data_set.has_person_named(person.name):
                    num_confusions += 1
                    print(f"MEH: There was confusion! Thought that {person.name} was {name} and let them in.")
                else:
                    num_imposters += 1
                    print(f"BAD: There was an imposter! Thought {person.name} was {name} and let them in")
        else:
            if test_data_set.has_person_named(person.name):
                num_betrayals += 1
                print(f"SAD: There was a betrayal! Could not recognize {person.name} and shut them out!")
    return f"\nbetrayals:{num_betrayals}\nConfusions:{num_confusions}\nImposters:{num_imposters}\ntotal:{num_vectors}\n" \
           f"Ratio:{num_vectors} / {(num_confusions + num_imposters + num_betrayals)}"


class Recognizer:
    def __init__(self, save=True, reset=True):
        self.save = save
        self.reset = reset
        self.captains_log = None
        self._set_up_logging()
        if self.save:
            self.captains_log.debug(f"Program starting, overwriting previous data is allowed. save ={self.save}")
        else:
            self.captains_log.debug(f"Program starting, overwriting previous data NOT allowed. save ={self.save}")
        self.base_data_set = self.get_data_set('base')
        self.celebrity_data_set = self.get_data_set('celebrity')
        self.test_data_set = None
        self.test_values = None

    def run_test(self):
        if self.test_data_set is None or self.test_values is None:
            raise ValueError('No test data to test!')
        method = run_test_helper
        for message in self.test_values.call_on_each_person(method, test_data_set=self.test_data_set):
            print(message)

    def make_test_data_set(self, name, directory='/test'):
        self.test_data_set = self.get_data_set(name, directory)

    def make_test_values(self, name, directory='/test'):
        self.test_values = self.get_test_values(name, directory)

    def is_in_base(self, image):
        return self.base_data_set.is_in_data_set(image)

    def closest_celebrity(self, image):
        return self.celebrity_data_set.closest_person_in_data_set(image)

    def get_data_set_by_name(self, str_name):
        if str_name == self.base_data_set.name:
            return self.base_data_set
        if str_name == self.celebrity_data_set.name:
            return self.celebrity_data_set

    def save_data_set(self, data_set):
        self.captains_log.debug(f'saving numpy array of length 1 with data_set {data_set.name} as only object')
        self.captains_log.debug(f"saving to {dir_path}numpy_arrays/{data_set.name}")
        np.save(f"{dir_path}numpy_arrays/{data_set.name}", np.array([data_set]))

    def get_data_set(self, name, directory='/data_sets'):
        data_set = self.get_saved_data_set(name)[0]
        if not data_set:
            if not self.reset:
                print('Could not find dataset. Creating new one')
                self.captains_log.error('Could not find saved dataset. Creating new one')
            data = DataSet(name, self.save, directory)
            for message in data.set_up():
                self._log_message(message[0], message[1])
            if self.save:
                self.save_data_set(data)
            return data
        else:
            if self.reset:
                data = DataSet(name, self.save, directory)
                for message in data.set_up():
                    self._log_message(message[0], message[1])
                if self.save:
                    self.save_data_set(data)
                return data
            return data_set

    def get_saved_data_set(self, name):
        try:
            array = np.load(f"{dir_path}numpy_arrays/{name}.npy", allow_pickle=True)
        except IOError:
            self.captains_log.error('person_list file is corrupted or missing')
            self.captains_log.error('continuing with an empty array')
            return [None]
        except ValueError:
            self.captains_log.error("Problem loading array. Perhaps there is a problem with pickle?")
            self.captains_log.error('continuing with an empty array')
            return [None]
        return array

    def get_test_values(self, name, directory='/test'):
        test_values = self.get_saved_test_values(name)[0]
        if not test_values:
            if not self.reset:
                print('Could not find test_values. Creating new ones')
                self.captains_log.error('Could not find saved test_values. Creating new ones')
            values = TestSet(name, directory)
            for message in values.set_up():
                self._log_message(message[0], message[1])
            if self.save:
                self.save_data_set(values)
            return values
        else:
            if self.reset:
                values = TestSet(name, directory)
                for message in values.set_up():
                    self._log_message(message[0], message[1])
                if self.save:
                    self.save_data_set(values)
                return values
            return test_values

    def get_saved_test_values(self, name):
        try:
            array = np.load(f"{dir_path}numpy_arrays/{name}.npy", allow_pickle=True)
        except IOError:
            self.captains_log.error('test_list file is corrupted or missing')
            self.captains_log.error('continuing with an empty array')
            return [None]
        except ValueError:
            self.captains_log.error("Problem loading array. Perhaps there is a problem with pickle?")
            self.captains_log.error('continuing with an empty array')
            return [None]
        return array

    def delete_processed_data(self, data_set):
        folder = f"{dir_path}processed_data/{data_set.name}"
        print(folder)
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            print(file_path)
            try:
                if os.path.isdir(file_path):
                    self._delete_inside(file_path)
            except Exception as e:
                self.captains_log.error(f"There has been an exception: {e}")
                print(e)

    def _delete_inside(self, folder):
        print(folder)
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            print(file_path)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                self.captains_log.error(f"There has been an exception trying to clear files.{e}")
                print(e)

    def _set_up_logging(self):
        logging.basicConfig(filename="captains_log/_captains.log", level=logging.NOTSET,
                            format='(%(levelname)s)) - %(asctime)s - %(message)s')
        self.captains_log = logging.getLogger('captains_log')

        handler = logging.handlers.RotatingFileHandler('captains_log/_captains.log', maxBytes=4096, backupCount=10)
        self.captains_log.addHandler(handler)

    def _log_message(self, message_type, message):
        if message_type == 'debug':
            self.captains_log.debug(message)
            return
        if message_type == 'error':
            self.captains_log.error(message)
            return
        if message_type == 'warning' or message_type == 'warn':
            self.captains_log.warning(message)
            return
        if message_type == 'info':
            self.captains_log.info(message)
            return
        self.captains_log.debug(f"Bad log message. type: {message_type}, message:{message}")


def main():
    ap = ArgumentParser()
    ap.add_argument("-l", "--reset", default="True", help="Do you want the data to be reloaded and overwrite "
                                                          "previous save?")
    ap.add_argument("-s", "--save", default="False", help="Do you want to rewrite the previous save data?")
    ap.add_argument('-t', "--test", default="True", help="Do you want to test the recognition?")
    ap.add_argument('-v', "--video", default='True', help="Do you want to see the video recognition?")
    args = vars(ap.parse_args())
    print(f"save={args['save'] in ['true', 'True', 't', 'T']}")
    print(f"reset={args['reset'] in ['true', 'True', 't', 'T']}")
    print(f"test={args['test'] in ['true', 'True', 't', 'T']}")
    print(f"video={args['video'] in ['true', 'True', 't', 'T']}")

    recognizer = Recognizer(args['save'] in ['true', 'True', 't', 'T'], args['reset'] in ['true', 'True', 't', 'T'])

    if args['test'] in ['true', 'True', 't', 'T']:
        run_test(recognizer)

    if args['video'] in ['true', 'True', 't', 'T']:
        capture_video(recognizer)
    return


def capture_video(recognizer):

    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)

    capture = cv.VideoCapture(0)
    aligner = Aligner()
    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        faces, boxes = aligner.align_image_get_boxes(frame)

        for i, box in enumerate(boxes):
            (start_x, start_y, end_x, end_y) = box.astype("int")
            image = faces[i]

            is_match, confidence, name = recognizer.is_in_base(image)

            celeb_closeness, celeb_name = recognizer.closest_celebrity(image)

            if is_match:
                cv.rectangle(frame, (start_x, start_y), (int(end_x * 1.1), int(end_y * 1.1)), green, 2)
                cv.putText(frame, f"%{round(confidence, 2)}: {name}", (start_x, start_y + 13),
                           cv.FONT_HERSHEY_COMPLEX, 0.45, green, 2)
            else:
                cv.rectangle(frame, (start_x, start_y), (int(end_x * 1.1), int(end_y * 1.1)), red, 2)
                cv.putText(frame, f"%{round(confidence, 2)}: {name}", (start_x, start_y + 13),
                           cv.FONT_HERSHEY_COMPLEX, 0.45, red, 2)

            cv.putText(frame, f"%{round(celeb_closeness, 2)}: {celeb_name}", (start_x, start_y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, blue, 2)

        frame = imutils.resize(frame, width=1200)
        cv.imshow("Frame", frame)
        key = cv.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            capture.release()
            cv.destroyAllWindows()
            break
    return


def run_test(recognizer):
    recognizer.make_test_data_set('base_small')
    recognizer.make_test_values('base_test')
    recognizer.run_test()


if __name__ == '__main__':
    main()
    exit()
