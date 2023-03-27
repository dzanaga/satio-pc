import azure.storage

from azure.storage.blob import ContainerClient

with open('../../../sastoken') as f:
    sas_token = f.read()
    
container_client = azure.storage.blob.ContainerClient(
    "https://dza2.blob.core.windows.net",
    container_name="feats",
    credential=sas_token,
) 

container_client.upload_blob(...)

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# Set the connection string, container name, and file path
with open('../../../connstr') as f:
    connect_str = f.read()
    
container_name = 'habitattest'

# Create a BlobServiceClient object
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# Create a ContainerClient object
container_client = blob_service_client.create_container(container_name)
# container_client = blob_service_client.get_container_client(container_name)

# Upload the file to Blob Storage
with open(fn, 'rb') as data:
    blob_client = container_client.upload_blob(name=fn, data=data)

print(f'The file {fn} has been uploaded to {container_name} container.')


# Download the blob to a local file
with open(blob_name, 'wb') as my_blob:
    download_stream = container_client.download_blob(blob_name)
    my_blob.write(download_stream.readall())
