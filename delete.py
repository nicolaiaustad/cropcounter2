import os
import shutil

def delete_all_files_in_directory(directory):
    """
    Delete all files in the specified directory.
    
    :param directory: The directory to delete files from.
    """
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted directory and its contents: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# Example usage
directory_path = "/home/dataplicity/remote"

delete_all_files_in_directory(directory_path)