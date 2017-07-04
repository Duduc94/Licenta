"""
Author: Duduc Ionut
Date: 05.04.2017
"""

import os
import re
import cv2
import glob
import argparse


class FrameExtractor:
    def __init__(self, arguments=None):
        """
        Constructor: sets current directory and parses arguments passed in terminal
        """
        self.curr_dir_path = os.path.abspath(os.path.dirname(__file__))
        self.args = arguments

    @staticmethod
    def process_file(path):
        """
        Process single video to extract all frames from it
        :param path: the absolute path to the video file
        :return: an array containing all the extracted frames
        """
        frames = []
        video_capture = cv2.VideoCapture(path)
        extracted, image = video_capture.read()
        while extracted:
            frames.append(image)
            extracted, image = video_capture.read()
        if len(frames):
            print("Successfully extracted %d frames from %s" % (len(frames), path))
        else:
            print("No frames extracted from %s" % path)

        return frames

    def extract(self):
        """
        Parse process entire directory if -d flag is specified, or just one file
        """
        dict = {'a': ['anger', 0], 'd': ['disgust', 0], 'f': ['fear', 0], 'h': ['happiness', 0],
                'n': ['neutral', 0], 'sa': ['sadness', 0], 'su': ['surprise', 0], 'total': ['', 0]}
        if self.args.directory:
            # parse entire directory and process all .avi files in it
            if not os.path.isabs(self.args.source):
                self.args.source = os.path.join(self.curr_dir_path, self.args.source)
                for file_path in glob.iglob(os.path.join(self.args.source, r'**/*.avi'), recursive=True):
                    frames = self.process_file(file_path)
                    if len(frames):
                        if not os.path.isabs(self.args.destination):
                            self.args.destination = os.path.join(self.curr_dir_path, self.args.destination)
                        for frame in frames:
                            if self.args.filter:
                                # frames need to be placed in specific folder, based on their label
                                regex = r"([a-zA-Z]+)\d+"
                                file_label = re.findall(regex, os.path.split(file_path)[-1])[0]
                                for key in dict.keys():
                                    if key == file_label:
                                        cv2.imwrite(os.path.join(self.args.destination, dict[key][0],
                                                                 "%s.%d.jpg" % (dict[key][0], dict[key][1])), frame)
                                        dict[key][1] += 1
                            else:
                                # frames are all placed in same folder, regardless of their label
                                cv2.imwrite(os.path.join(self.args.destination, "%d.jpg" % dict['total'][1]), frame)
                                dict['total'][1] += 1
        else:
            # process single file
            frames = self.process_file(os.path.join(self.curr_dir_path, self.args.source))
            if len(frames):
                if not os.path.isabs(self.args.destination):
                    self.args.destination = os.path.join(self.curr_dir_path, self.args.destination)
                no_of_frames = 0
                for frame in frames:
                    cv2.imwrite(os.path.join(self.args.destination, "%d.jpg" % no_of_frames), frame)
                    no_of_frames += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filter", action="store_true", help="specifies that frames should be saved "
                        "in different subdirectories, based on their name")
    parser.add_argument("-d", "--directory", action="store_true", help="specifies the source is a directory")
    parser.add_argument("source", help="source video file or directory")
    parser.add_argument("destination", help="destination directory for frames")
    args = parser.parse_args()

    frameExtractor = FrameExtractor(args)
    frameExtractor.extract()
