import modal
import os
import shutil

stub = modal.Stub("transformer")
volume = modal.NetworkFileSystem.new().persisted("transformer")

@stub.function(mounts=[modal.Mount.from_local_dir("./input", remote_path="/source/input")], network_file_systems={"/root/content": volume})
def load_dataset():
    source_dataset_path = "/source/input"
    dest_dataset_path = "/root/content/data"

    def check():        
        if os.path.exists(dest_dataset_path):
            files = os.listdir(dest_dataset_path)
            print(f"File exists: {str.join(', ', files)}")
        else:
            print(f"Path doesn't exist")
    check()
    shutil.copytree(source_dataset_path, dest_dataset_path, dirs_exist_ok=True)
    print("files copied")
    check()

# Create Bigram Mode

@stub.local_entrypoint()
def main():
    load_dataset.remote()



