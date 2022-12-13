import csv
import itertools
import os
import collections.abc as collections
from abc import ABC
import json
import numpy as np
import torch
from tqdm import tqdm

from pc_io.read_save import load_obj
from utils import dict_to_ordered_dict


class DatasetManager(collections.Iterator, ABC):
    def __init__(self, dataset_dir, cache_labels=False, cache_centroids=False, cache_point_clouds=False):
        self.dataset_dir = dataset_dir
        self.metadata_dir = os.path.join(self.dataset_dir, "metadata")
        self.dataset_name = dataset_dir.split("/")[-1]

        self.train = ItemManager(os.path.join(self.metadata_dir, "train-scenes.csv"), os.path.join(self.metadata_dir, "train-metadata.json"), self.dataset_dir, cache_labels=False, cache_centroids=False, cache_point_clouds=False)
        self.test = ItemManager(os.path.join(self.metadata_dir, "test-scenes.csv"), os.path.join(self.metadata_dir, "train-metadata.json"), self.dataset_dir, cache_labels=False, cache_centroids=False, cache_point_clouds=False)
        self.val = ItemManager(os.path.join(self.metadata_dir, "val-scenes.csv"), os.path.join(self.metadata_dir, "val-metadata.json"), self.dataset_dir, cache_labels=False, cache_centroids=False, cache_point_clouds=False)

        self._count = None

        self.cache_labels = cache_labels
        self.cache_centroids = cache_centroids
        self.cache_point_clouds = cache_point_clouds

        self._mask = np.ones(shape=(self.count()))

    def filter_clouds_with_double_centroids(self):
        self.train.filter_clouds_with_double_centroids()
        self.test.filter_clouds_with_double_centroids()
        self.val.filter_clouds_with_double_centroids()
        self._reindex()

    def custom_filter(self, filter):
        self.train.custom_filter(filter)
        self.test.custom_filter(filter)
        self.val.custom_filter(filter)
        self._reindex()

    def filter_missing_data(self):
        self.train.filter_missing_data()
        self.test.filter_missing_data()
        self.val.filter_missing_data()
        self._reindex()

    def filter_all_ones(self):
        self.train.filter_all_ones()
        self.test.filter_all_ones()
        self.val.filter_all_ones()
        self._reindex()

    def filter_scenes_with_zero_counts(self):
        self.train.filter_scenes_with_zero_counts()
        self.test.filter_scenes_with_zero_counts()
        self.val.filter_scenes_with_zero_counts()
        self._reindex()

    def filter_bad_geometries(self, bad_geometries):
        self.train.filter_bad_geometries(bad_geometries)
        self.test.filter_bad_geometries(bad_geometries)
        self.val.filter_bad_geometries(bad_geometries)
        self._reindex()

    def count(self, force_refresh=False):
        if self._count and not force_refresh:
            return self._count
        else:
            self._count = self.train.count(force_refresh=force_refresh) + self.test.count(force_refresh=force_refresh) + self.val.count(force_refresh=force_refresh)

        return self._count

    def set_cache_policy(self, cache_labels=False, cache_centroids=False, cache_point_clouds=False):
        self.train.set_cache_policy(cache_labels=cache_labels, cache_centroids=cache_centroids, cache_point_clouds=cache_point_clouds)
        self.test.set_cache_policy(cache_labels=cache_labels, cache_centroids=cache_centroids, cache_point_clouds=cache_point_clouds)
        self.val.set_cache_policy(cache_labels=cache_labels, cache_centroids=cache_centroids, cache_point_clouds=cache_point_clouds)

    def list_all_geo(self):
        for i, item in enumerate(self):
            print(i, item, set(map(lambda count_dict: count_dict["object_type"], item.label.slots.values())))

    def list_all_counts(self):
        for i, item in enumerate(self):
            print(i, item, set(map(lambda count_dict: count_dict["count"], item.label.slots.values())))

    def _reindex(self):
        self.count(force_refresh=True)


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"<{self.dataset_name} ({self.count()} items)>"

    def __len__(self):
        return self.count()

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index < self.train.count():
            i = self._iter_index
            self._iter_index += 1
            return self.train[i]
        elif self._iter_index < self.train.count() + self.test.count():
            i = self._iter_index
            self._iter_index += 1
            return self.test[i - self.train.count()]
        elif self._iter_index < self.train.count() + self.test.count() + self.val.count():
            i = self._iter_index
            self._iter_index += 1
            return self.val[i - self.train.count() - self.test.count()]
        else:
            raise StopIteration

    def __getitem__(self, index):
        if index < self.train.count():
            return self.train[index]
        elif index < self.train.count() + self.test.count():
            return self.test[index - self.train.count()]
        else:
            return self.val[index - self.train.count() - self.test.count()]



class ItemManager(collections.Iterator, ABC):
    def __init__(self, scenes_file, metadata_file, dataset_dir, cache_labels=False, cache_centroids=False, cache_point_clouds=False):
        self.scenes_file = scenes_file
        self.metadata_file = metadata_file
        self.dataset_dir = dataset_dir

        self._count = None
        self._mask = None
        self._items_array = [None] * self.count()
        self._iter_index = 0
        self._mask = np.ones(shape=(self.count()))
        self._index_to_line_number = np.arange(self.count())

        self.cache_labels = cache_labels
        self.cache_centroids = cache_centroids
        self.cache_point_clouds = cache_point_clouds

        self._geometry_map = None
        self._normalizer = None

    def custom_filter(self, filter):
        new_mask = np.ones(shape=self.count())
        
        for i, item in tqdm(enumerate(self), total=len(self)):
            new_mask[i] = filter(item)

        self._reindex(new_mask)

    def filter_missing_data(self):
        new_mask = np.ones(shape=self.count())
        
        for i, item in tqdm(enumerate(self), total=len(self)):
            if not os.path.isfile(item.obj.path):
                new_mask[i] = 0
            if not os.path.isfile(item.label.path):
                new_mask[i] = 0

        self._reindex(new_mask)

    def filter_clouds_with_double_centroids(self):
        new_mask = np.ones(shape=self.count())

        print("Total clouds:", sum(new_mask))
        for i, item in tqdm(enumerate(self), total=len(self)):
            points = [x['points'] for x in item.centroids.data.values()]
            points.reverse()
            for j in range(0, len(points) - 1):
                diff = np.array(points[j]) - np.array(points[j+1])
                # print(diff[0] - diff[2])
                if diff[0] - diff[2] == 0:
                    new_mask[i] = 0
                    print(item.obj.path)
                    
                    break

        print("Total good clouds:", sum(new_mask))

        # self._reindex(new_mask)

    def filter_all_ones(self):
        new_mask = np.ones(shape=self.count())
        for i, item in tqdm(enumerate(self), total=len(self)):
            if len(item.label.slot_counts.unique()) == 1 and 1 in item.label.slot_counts:
                new_mask[i] = 0

        self._reindex(new_mask)

    def filter_scenes_with_zero_counts(self):
        new_mask = np.ones(shape=self.count())
        for i, item in tqdm(enumerate(self), total=len(self)):
            if 0 in item.label.slot_counts:
                new_mask[i] = 0

        self._reindex(new_mask)

    def filter_bad_geometries(self, bad_geometries):
        new_mask = np.ones(shape=self.count())
        for i, item in tqdm(enumerate(self), total=len(self)):
            unique_geo = set(map(lambda count_dict: count_dict["object_type"], item.label.slots.values()))
            if len(unique_geo.intersection(set(bad_geometries))) > 0:
                new_mask[i] = 0

        self._reindex(new_mask)

    def list_all(self):
        for i, item in enumerate(self):
            print(i, item, set(map(lambda count_dict: count_dict["count"], item.label.slots.values())))

    def count(self, force_refresh=False):
        if self._count and not force_refresh:
            return self._count
        else:
            if hasattr(self, "_mask") and self._mask is not None:
                return int(sum(self._mask))
            else:
                return int(sum(1 for _ in open(self.scenes_file)))

    def set_cache_policy(self, cache_labels=False, cache_centroids=False, cache_point_clouds=False):
        self.cache_lablels = cache_labels
        self.cache_centroids = cache_centroids
        self.cache_point_clouds = cache_point_clouds

    def _get_line(self, line_number):
        with open(self.scenes_file) as f:
            return next(itertools.islice(csv.reader(f), line_number, None))

    def _reindex(self, new_mask):
        for i, val in enumerate(new_mask):
            if val == 0:
                self._mask[self._index_to_line_number[i]] = 0

        self.count(force_refresh=True)
        self._index_to_line_number = np.zeros(self.count(), dtype=int)
        actual_index = 0
        for i, val in enumerate(self._mask):
            if val == 1:
                self._index_to_line_number[actual_index] = i
                actual_index += 1

    def _get_metadata(self):
        if self._geometry_map and self._normalizer:
            return self._geometry_map, self._normalizer
        else:
            with open(self.metadata_file) as label_file:
                data = json.load(label_file)
                self._geometry_map, self._normalizer = data["geometry_map"], data["normalizer"]
                return self._geometry_map, self._normalizer

    def __getitem__(self, index):
        line_index = self._index_to_line_number[index]
        if self._items_array[line_index]:
            return self._items_array[line_index]
        else:
            item = Item(self._get_line(line_index), self.dataset_dir, cache_labels=self.cache_labels, cache_centroids=self.cache_centroids, cache_point_clouds=self.cache_point_clouds)
            self._items_array[line_index] = item
            return self._items_array[line_index]

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index < self.count():
            i = self._iter_index
            self._iter_index += 1
            return self.__getitem__(i)
        else:
            raise StopIteration

    def __getattr__(self, attr):
        if attr == "geomap":
            return self._get_metadata()[0]
        elif attr == "normalizer":
            return self._get_metadata()[1]

    def __len__(self):
        return self.count()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"<ItemManager ({self.count()} items)>"


class Item(object):
    def __init__(self, item_info_list, dataset_dir, cache_labels=False, cache_centroids=False, cache_point_clouds=False):
        self.item_info_list = item_info_list
        self.dataset_dir = dataset_dir

        label_local_path, obj_local_path, centroid_local_path, source, images_list = self.item_info_list

        self.label = Label(os.path.join(self.dataset_dir, label_local_path), cache=cache_labels)
        self.obj = Obj(os.path.join(self.dataset_dir, obj_local_path), cache=cache_point_clouds)
        self.centroids = Centroid(os.path.join(self.dataset_dir, centroid_local_path), cache=cache_centroids)

        self.source = source

        images_paths = json.loads(images_list.replace("'", '"'))
        self.images = [Image(os.path.join(self.dataset_dir, x)) for x in images_paths]

        self.id = label_local_path.split("/")[-1][0:36]

    def _get_details(self):
        return {
            "id": self.id,
            "source": self.source,
            "label": self.label.path,
            "obj": self.obj.path,
            "images": self.images
        }

    def __getattr__(self, attr):
        if attr == "details":
            return self._get_details()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"<Item {self.id}>"


class Label(object):
    def __init__(self, label_file, cache):
        self._label_file = label_file
        self._local_label_file = "/".join(self._label_file.split("/")[-2:])

        self._slots = None
        self._slot_counts = None

        self._cache = cache

    def _get_slots(self):
        if self._slots:
            return self._slots
        else:
            with open(self._label_file) as label_file:
                l = json.load(label_file)
                if self._cache:
                    self._slots = l["slot_counts"]

                return l["slot_counts"]

    def _get_slot_counts(self):
        if self._slot_counts is not None:
            return self._slot_counts
        else:
            slots = self._get_slots()
            s_c = torch.tensor([s['count'] for s in slots.values()]).long()
            if self._cache:
                self._slot_counts = s_c
            return s_c




    def __getattr__(self, attr):
        if attr == "slots":
            return self._get_slots()
        if attr == "slot_counts":
            return self._get_slot_counts()
        if attr == "path":
            return self._label_file

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"<Label {self._local_label_file}>"


class Centroid(object):
    def __init__(self, centroid_file, cache):
        self._centroid_file = centroid_file
        self._local_centroid_file = "/".join(self._centroid_file.split("/")[-2:])
        self._cache = cache

        self._centroids = None

    def _get_centroid_data(self):
        if self._centroids:
            return self._centroids 
        else:
            with open(self._centroid_file) as centroid_file:
                c = json.load(centroid_file)
                ordered_dict = dict_to_ordered_dict(c["centroids"])
                if self._cache:
                    self._centroids = ordered_dict
                return ordered_dict
    
    

    def __getattr__(self, attr):
        if attr == "path":
            return self._centroid_file
        elif attr == "data":
            return self._get_centroid_data()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"<Centroid {self._local_label_file}>"


class Obj(object):
    def __init__(self, obj_file, cache):
        self._obj_file = obj_file
        self._local_obj_file = "/".join(self._obj_file.split("/")[-2:])
        self._cache = cache

        self.path = self._obj_file

        self._num_points = None
        self._verts = None
        self._normals = None

    def _get_num_points(self):
        if self._num_points:
            return self._num_points
        else:
            with open(self._obj_file) as obj_file:
                counter = 0
                for line in obj_file:
                    if line[0].lower() in ["v", "p"]:
                        counter += 1
            self._num_points = counter
            return self._num_points

    def _get_points_verts(self):
        if self._verts is not None and self._normals is not None:
            return self._verts, self._normals
        else:
            v, n = load_obj(self._obj_file)
            if self._cache:
                self._verts, self._normals = v, n
            return v, n

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"<Obj {self._local_obj_file}>"

    def __getattr__(self, attr):
        if attr == "num_points":
            return self._get_num_points()
        if attr == "path":
            return self._obj_file
        if attr == "data":
            return self._get_points_verts()


class Image(object):
    def __init__(self, image_file):
        self._image_file = image_file
        self._local_image_file = "/".join(self._image_file.split("/")[-2:])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"<Image {self._image_file} >"

    def get_path(self):
        return self._image_file
