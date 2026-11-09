"""Archived removal utility. Deletes only a selected ignored local-data folder."""
import shutil

from . import config
from .train import train_model


def list_people():
    if not config.DATASET_PATH.exists():
        return []
    return sorted(path.name for path in config.DATASET_PATH.iterdir() if path.is_dir())


def remove_person(name):
    person_dir = config.DATASET_PATH / name
    if not person_dir.exists():
        print(f"No such person: '{name}'")
        return False
    shutil.rmtree(person_dir)
    print(f"Removed '{name}' from dataset.")
    return True


if __name__ == "__main__":
    people = list_people()
    if not people:
        print("Dataset is empty — nothing to remove.")
    else:
        print("Currently enrolled:", ", ".join(people))
        if remove_person(input("Enter the name to remove: ").strip()):
            print("Retraining model without this person...")
            train_model()
