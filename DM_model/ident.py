import os
import hashlib

def hash_file(file_path):
    """Generate a hash for a file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def count_identical_files(directory):
    """Count the number of identical files in the given directory."""
    files_hash = {}
    identical_count = 0

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_hash = hash_file(file_path)

            if file_hash in files_hash:
                identical_count += 1
            else:
                files_hash[file_hash] = file_path

    print(len(files_hash))
    return identical_count

# Example usage
directory = '/home/andrea/Scrivania/universita/Probabilistic ML/DL_project-main/vae_chris/root/GenetatedDM'
identical_count = count_identical_files(directory)
