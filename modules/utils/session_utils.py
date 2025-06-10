# modules/session_utils.py

import pickle
import numpy as np


def store_digit_images(session, digit_images):
    session["digit_images"] = pickle.dumps(digit_images)


def load_digit_images(session):
    return pickle.loads(session["digit_images"])


def store_board(session, board: np.ndarray):
    session["unsolved_board"] = board.tolist()


def load_board(session):
    return np.array(session["unsolved_board"])


def store_filename(session, filename):
    if isinstance(filename, str) and filename.strip():
        session["filename"] = filename.strip()


def load_filename(session):
    return session.get("filename")
