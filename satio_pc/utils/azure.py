from pathlib import Path
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


class AzureBlobReader:

    def __init__(self, connect_str, container_name):
        self._connect_str = connect_str
        self._container_name = container_name

    @property
    def blob_service_client(self):
        return BlobServiceClient.from_connection_string(self._connect_str)

    @property
    def container_client(self):
        return self.blob_service_client.get_container_client(
            self._container_name)
     
    def list_folders(self, prefix=None):
        folders = set()
        for blob in self.container_client.list_blobs(name_starts_with=prefix):
            folders.add(str(Path(blob.name).parent))
        return sorted(list(folders))

    def list_files(self, prefix=None):
        files = set()
        for blob in self.container_client.list_blobs(name_starts_with=prefix):
            if blob.name[-1] != "/":
                files.add(blob.name)
        return sorted(list(files))

    def check_file_exists(self, file_name):
        blob_client = self.container_client.get_blob_client(file_name)
        return blob_client.exists()

    def upload_folder(self, src_folder_path, dst_folder_name):
        src_folder_path = Path(src_folder_path)
        dst_folder_name = Path(dst_folder_name)
        for path in src_folder_path.rglob("*"):
            if path.is_file():
                src_file_path = path
                dst_file_name = dst_folder_name.joinpath(
                    path.relative_to(src_folder_path)).as_posix()
                self.upload_file(src_file_path, dst_file_name)

    def upload_file(self, src_file_path, dst_file_name, overwrite=False):
        blob_client = self.container_client.get_blob_client(dst_file_name)
        if not overwrite and blob_client.exists():
            return
        with open(src_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=overwrite)

    def delete_folder(self, folder_name):
        for blob in self.container_client.list_blobs(
                name_starts_with=folder_name):
            self.container_client.delete_blob(blob.name)

    def delete_file(self, file_name):
        if not self.check_file_exists(file_name):
            return
        self.container_client.delete_blob(file_name)

    def read_txt(self, blob_name):
        blob_client = self.container_client.get_blob_client(blob_name)
        if blob_client.content_settings.content_type != "text/plain":
            raise ValueError("The blob is not a text file")
        return blob_client.download_blob_to_text()
    
    def download_file(self, key, dst_file_path, overwrite=False):
        blob_client = self.container_client.get_blob_client(key)
        if not blob_client.exists():
            raise ResourceNotFoundError(f"The blob '{key}' does not exist")
        if not overwrite and Path(dst_file_path).exists():
            return
        with open(dst_file_path, "wb") as download_file:
            blob_data = blob_client.download_blob()
            blob_data.readinto(download_file)
