from google.cloud import storage
from google.api_core.exceptions import Forbidden, NotFound
import json
from collections import OrderedDict
import yaml
import os


def upload_blob(bucket_name, source_file_name, destination_blob_name, auth, overwrite = False):
    """Uploads a file to the bucket."""
    #storage_client = storage.Client()
    storage_client = storage.Client.from_service_account_json(auth)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    if blob.exists() and not overwrite:
        return
    else:
        try:
            blob.upload_from_filename(source_file_name, timeout=600)
        except Exception:
            print("Error uploading file using timeout, uploading without timeout")
            blob.upload_from_filename(source_file_name)


def copy_blob(
    bucket_name, blob_name, destination_bucket_name, destination_blob_name, auth=None):
    """Copies a blob from one bucket to another with a new name."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"
    # destination_bucket_name = "destination-bucket-name"
    # destination_blob_name = "destination-object-name"

    if auth:
        storage_client = storage.Client.from_service_account_json(auth)
    else:
        storage_client = storage.Client()

    source_bucket = storage_client.bucket(bucket_name)
    source_blob = source_bucket.blob(blob_name)
    destination_bucket = storage_client.bucket(destination_bucket_name)

    blob_copy = source_bucket.copy_blob(
        source_blob, destination_bucket, destination_blob_name
    )

    print(
        "Blob {} in bucket {} copied to blob {} in bucket {}.".format(
            source_blob.name,
            source_bucket.name,
            blob_copy.name,
            destination_bucket.name,
        )
    )

def download_blob(bucket_file_name, destination_file_name, skip_if_present=False, auth=None, verbose=True, fail_if_missing=False):
    """Downloads a blob from the bucket."""
    if not skip_if_present or not os.path.exists(destination_file_name):
        if auth:
            storage_client = storage.Client.from_service_account_json(auth)
        else:
            storage_client = storage.Client()
        bucket_file_name = bucket_file_name if bucket_file_name.startswith("gs://") else f"gs://{bucket_file_name}"
        blob = storage.blob.Blob.from_string(bucket_file_name, storage_client)
        bucket = storage_client.get_bucket(bucket_file_name.split("/")[2])

        b = "/".join(bucket_file_name.split("/")[3:])
        blob_file = bucket.get_blob(b)


        if blob_file:
            try:
                with open(destination_file_name, 'wb') as destination_file:
                    blob.download_to_file(destination_file, storage_client)
            except Forbidden:
                raise Forbidden(f"Forbidden access file on GCP at {bucket_file_name}.")
            except NotFound:
                print("Missing") #raise NotFound(f"Resource {bucket_file_name} not found on bucket {blob.bucket.name}")
        else:
            if verbose:
                print(f"Skipped: {destination_file_name}")
            if fail_if_missing:
                raise NotFound(f"Resource {bucket_file_name} not found on bucket {blob.bucket.name}")

    elif verbose:
        print(f"Skipped: {destination_file_name}")

def print_yaml(cfg):
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for k2, v2 in v.items():
                print(f"\t - {k2}: {v2}")
        else:
            print(f"{k}: {v}")


def read_json(fpath):
    with open(fpath, "r") as f:
        d = json.load(f)
    return d

def write_json(fpath, labels):
    with open(fpath, "w") as fout:
        json.dump(labels, fout)


def get_n_params(model):
    n_params = 0
    for layer in model.parameters():
        dims = layer.size()
        cnt = dims[0]
        for d in dims[1:]:
            cnt *= d
        n_params += cnt

    return n_params


def get_yaml(fpath, silent=False):
    with open(fpath, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if not silent:
        print("TRAINING SETTINGS:")
        print_yaml(cfg)
    return cfg


def dict_to_ordered_dict(d):
    ordered = OrderedDict()
    for k, v in d.items():
        ordered[k] = v
    return ordered