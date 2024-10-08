import numpy as np
import cv2
import mindspore.dataset as ds
from mindspore import Tensor
# import matplotlib.pyplot as plt
from PIL import Image

cv2.setNumThreads(0)


class SegDataset:
    def __init__(self,
                 data_file='',
                 batch_size=32,
                 image_size=512,
                 image_mean=[86.588776, 89.02508, 71.802475],
                 image_std=[28.061657, 27.053534, 30.217733],
                 max_scale=2.0,
                 min_scale=0.5,
                 num_readers=2,
                 num_parallel_calls=4,
                 shard_id=None,
                 shard_num=None):

        self.data_file = data_file
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.num_readers = num_readers
        self.num_parallel_calls = num_parallel_calls
        self.shard_id = shard_id
        self.shard_num = shard_num
        assert max_scale > min_scale

    # def preprocess_(self, data, mask):
    #     # BGR image
    #     # image_1 = cv2.imdecode(np.frombuffer(augment_data_1, dtype=np.int8), cv2.IMREAD_COLOR)
    #     # image_2 = cv2.imdecode(np.frombuffer(augment_data_2, dtype=np.int8), cv2.IMREAD_COLOR)
    #     image = cv2.imdecode(np.frombuffer(data, dtype=np.int8), cv2.IMREAD_COLOR)
    #     image = (image - self.image_mean) / self.image_std
    #     image = image.transpose((2, 0, 1))
    #     mask = cv2.imdecode(np.frombuffer(mask, dtype=np.int8), cv2.IMREAD_GRAYSCALE)
    #     # Normalize
    #     # image_1 = (image_1 - self.image_mean) / self.image_std
    #     # image_2 = (image_2 - self.image_mean) / self.image_std
    #
    #     # image_1 = image_1.transpose((2, 0, 1))
    #     # image_2 = image_2.transpose((2, 0, 1))
    #     # image1 = image_1.copy()
    #     # image2 = image_2.copy
    #     return image, mask


    #
    # def get_dataset(self, repeat=1):
    #     data_set = ds.MindDataset(dataset_files=self.data_file, columns_list=["data", "mask"],
    #                               shuffle=True, num_parallel_workers=self.num_readers,
    #                               num_shards=self.shard_num, shard_id=self.shard_id)
    #     transforms_list = self.preprocess_
    #     data_set = data_set.map(operations=transforms_list, input_columns=["data", "mask"],
    #                             output_columns=["data", "mask"],
    #                             num_parallel_workers=self.num_parallel_calls)
    #     # data_set = data_set.shuffle(buffer_size=self.batch_size * 10)
    #     data_set = data_set.batch(self.batch_size, drop_remainder=True)
    #     data_set = data_set.repeat(repeat)
    #     return data_set
    def preprocess_(self, augment_data_1, mask):
        # BGR image
        # image_1 = cv2.imdecode(np.frombuffer(augment_data_1, dtype=np.int8), cv2.IMREAD_COLOR)
        # image_2 = cv2.imdecode(np.frombuffer(augment_data_2, dtype=np.int8), cv2.IMREAD_COLOR)
        image = cv2.imdecode(np.frombuffer(augment_data_1, dtype=np.int8), cv2.IMREAD_COLOR)
        image = (image - self.image_mean) / self.image_std
        image = image.transpose((2, 0, 1))
        mask = cv2.imdecode(np.frombuffer(mask, dtype=np.int8), cv2.IMREAD_GRAYSCALE)
        # Normalize
        # image_1 = (image_1 - self.image_mean) / self.image_std
        # image_2 = (image_2 - self.image_mean) / self.image_std

        # image_1 = image_1.transpose((2, 0, 1))
        # image_2 = image_2.transpose((2, 0, 1))
        # image1 = image_1.copy()
        # image2 = image_2.copy
        return image, mask

    def get_dataset(self, repeat=1):
        data_set = ds.MindDataset(dataset_files=self.data_file, columns_list=["augment_data_1", "mask"],
                                  shuffle=True, num_parallel_workers=self.num_readers,
                                  num_shards=self.shard_num, shard_id=self.shard_id)
        transforms_list = self.preprocess_
        data_set = data_set.map(operations=transforms_list, input_columns=["augment_data_1", "mask"],
                                output_columns=["data", "mask"],
                                num_parallel_workers=self.num_parallel_calls)
        # data_set = data_set.shuffle(buffer_size=self.batch_size * 10)
        data_set = data_set.batch(self.batch_size, drop_remainder=True)
        data_set = data_set.repeat(repeat)
        return data_set


    def get_dataset_eval(self, repeat=1):
        data_set = ds.MindDataset(dataset_files=self.data_file, columns_list=["augment_data_1", "augment_data_2", "mask"],
                                  shuffle=True, num_parallel_workers=self.num_readers,
                                  num_shards=self.shard_num, shard_id=self.shard_id, num_samples=200)
        transforms_list = self.preprocess_
        data_set = data_set.map(operations=transforms_list, input_columns=["augment_data_1", "augment_data_2", "mask"],
                                output_columns=["augment_data_1", "augment_data_2", "mask"],
                                num_parallel_workers=self.num_parallel_calls)
        # data_set = data_set.shuffle(buffer_size=self.batch_size * 10)
        data_set = data_set.batch(self.batch_size, drop_remainder=True)
        data_set = data_set.repeat(repeat)
        return data_set

    # def preprocess(self, augment_data_1):
    #     # BGR image
    #     image_1 = cv2.imdecode(np.frombuffer(augment_data_1, dtype=np.int8), cv2.IMREAD_COLOR)
    #     # Normalize
    #     image_1 = (image_1 - self.image_mean) / self.image_std
    #     image_1 = image_1.transpose((2, 0, 1))
    #
    #     # image_size = 512
    #     # patch_size = 4
    #     # mask_size = 16
    #     # mask_ratio = 0.6
    #     #
    #     # rand_size = image_size // mask_size
    #     # scale = mask_size // patch_size
    #     #
    #     # token_count = rand_size ** 2
    #     # mask_count = int(np.ceil(token_count * mask_ratio))
    #     # # print(mask_count)
    #     #
    #     # mask_idx = np.random.permutation(token_count)[0:mask_count]
    #     # mask = np.zeros(token_count, dtype=int)
    #     # mask[mask_idx] = 1
    #     #
    #     # mask = mask.reshape((rand_size, rand_size))
    #     # mask = Tensor(mask.repeat(scale, axis=0).repeat(scale, axis=1))
    #     return image_1
    #
    # def get_dataset(self, repeat=1):
    #     data_set = ds.MindDataset(dataset_files=self.data_file, columns_list=["augment_data_1"],
    #                               shuffle=False, num_parallel_workers=self.num_readers,
    #                               num_shards=self.shard_num, shard_id=self.shard_id, num_samples=300)
    #     transforms_list = self.preprocess_
    #     data_set = data_set.map(operations=transforms_list, input_columns=["augment_data_1"],
    #                             output_columns=["image_1"],
    #                             column_order=["image_1"],
    #                             num_parallel_workers=self.num_parallel_calls)
    #     # data_set = data_set.shuffle(buffer_size=self.batch_size * 10)
    #     data_set = data_set.batch(self.batch_size, drop_remainder=True)
    #     data_set = data_set.repeat(repeat)
    #     return data_set
