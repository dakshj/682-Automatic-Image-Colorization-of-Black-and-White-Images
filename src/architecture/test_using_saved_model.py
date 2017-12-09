import os

from architecture.model import test
from data.io import get_project_dirs, load_model_from_disk

if __name__ == '__main__':
    # Go to project root directory
    os.chdir('..')
    os.chdir('..')

    train_dir, log_dir, model_dir, test_dir, colorized_dir = \
        get_project_dirs(project_root_dir=os.getcwd())

    model = load_model_from_disk(model_dir)
    test(model, test_dir=test_dir, colorized_dir=colorized_dir)
