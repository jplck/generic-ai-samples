from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.storage.blob import BlobLeaseClient
from dataclasses import dataclass
from pathlib import Path
import os

class Blob:
    
    container: str
    name: str
    lease_id: str

    def __init__(self, storage_url: str, credential: DefaultAzureCredential, container_name: str, blob_name: str):
        self.blob_service_client = BlobServiceClient(account_url=storage_url, credential=credential)
        self.container = container_name
        self.name = blob_name
        self.lease_id = None

    def download(self, destination: str):
        blob_data = self.get_blob_client().download_blob().readall()
        with open(destination, "wb") as f:
            f.write(blob_data)

    def lease(self):
        lease_client = BlobLeaseClient(self.get_blob_client())
        lease_client.acquire()
        self.lease_id = lease_client.id

    def is_locked(self) -> bool:
        return self.get_blob_client().get_blob_properties().lease.status == 'locked'

    def move_blob(self, target_container_name: str) -> Path:
        origin_blob_client = self.get_blob_client()
        target_blob_client = self.blob_service_client.get_blob_client(target_container_name, self.name)
        target_blob_client.start_copy_from_url(origin_blob_client.url)
        origin_blob_client.delete_blob(lease=self.lease_id)
        return target_blob_client.url

    def release_lease(self):
        lease_client = BlobLeaseClient(self.get_blob_client())
        lease_client.release()

    def get_blob_client(self):
        return self.blob_service_client.get_blob_client(self.container, self.name)
    
class Container:
    def __init__(self, storage_url: str, credential: DefaultAzureCredential, container_name: str):
        self.blob_service_client = BlobServiceClient(account_url=storage_url, credential=credential)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.container_name = container_name
        self.credentials = credential
        self.storage_url = storage_url

    def exists(self) -> bool:
        return self.container_client.exists()

    def create_container(self):
        if self.exists():
            return
        self.container_client.create_container()

    def get_files(self) -> list[Blob]:
        list = self.blob_service_client.get_container_client(self.container_name).list_blobs()
        return [Blob(self.storage_url, self.credentials, blob.container, blob.name) for blob in list]
    
    def upload_from_local(self, local_folder_path: str, remote_folder_name: str, metadata: dict):
        for root, dirs, files in os.walk(local_folder_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_folder_path)
                blob_name = f"{remote_folder_name}/{relative_path.replace('\\', '/')}"
                with open(local_file_path, "rb") as data:
                    self.blob_service_client.get_blob_client(self.container_name, blob_name).upload_blob(data, overwrite=True, metadata=metadata)
