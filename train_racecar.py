#import raceResnet
import racePilotNet
import tensorflow as tf
import time
import os
import sys
import re
import numpy as np
from termcolor import colored, cprint
from gtav.data_utils import get_dataset

from image_processing import image_preprocessing

def main(_):
    racePilotNet.train()


if __name__ == '__main__':
    tf.app.run()
