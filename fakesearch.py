import pickle


def load(picklepath):
    with open(picklepath, "rb") as f:
        return pickle.load(f)
