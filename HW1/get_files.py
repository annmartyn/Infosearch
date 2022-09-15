import os
from lemmatizer import get_lemmatized_dict

def open_and_lemmatize():
    all_files_paths = []
    files = []
    for root, dirs, path in os.walk('./friends-data'):
        for elem in path:
            all_files_paths.append(root + '/' + elem)
    current_dir = os.getcwd()
    for file_path in all_files_paths:
        files.append(os.path.join(current_dir, file_path))
    return get_lemmatized_dict(files)