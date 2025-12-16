import threading
import unittest

import numpy as np
import torch

from recis.data.local_data_sampler import (
    LocalDataSampler,
    ragged_char_to_string,
    string_to_ragged_char,
)
from recis.ragged.tensor import RaggedTensor


class LocalDataSamplerTest(unittest.TestCase):
    def mock_sample_tag_tensors(self):
        """
        [1]
        [1]
        """
        return RaggedTensor(
            values=torch.tensor([1, 1]), offsets=[torch.tensor([0, 1, 2])]
        )

    def mock_dedup_tag_tensors(self):
        """
        [0]
        [1]
        """
        return RaggedTensor(
            values=torch.tensor([0, 1]), offsets=[torch.tensor([0, 1, 2])]
        )

    def mock_batch(self):
        batch_size = 10
        sample_tag = torch.concat(
            [
                torch.ones(batch_size // 2, dtype=torch.int32),
                torch.zeros(batch_size // 2, dtype=torch.int32),
            ]
        )
        columns = {
            "sample_tag": RaggedTensor(
                values=sample_tag, offsets=[torch.arange(0, batch_size + 1)]
            ),
            "dedup_tag": RaggedTensor(
                values=torch.arange(0, batch_size),
                offsets=[torch.arange(0, batch_size + 1)],
            ),
            "weight_tag": torch.ones(batch_size, 1, dtype=torch.float32),
            "skey_name": [
                np.array([str(i).encode("ascii") * (i + 1) for i in range(batch_size)])
            ],
            "data": RaggedTensor(
                values=torch.arange(0, batch_size),
                offsets=[torch.arange(0, batch_size + 1)],
            ),
        }
        batch = [columns]
        return batch, batch_size

    def test_string_to_ragged_char(self):
        arr = np.array([b"1", b"23", b"456"], dtype="|S3")
        ragged_char = string_to_ragged_char(arr)
        values = ragged_char.values()
        splits = ragged_char.offsets()[0]
        expect_values = torch.tensor([49, 50, 51, 52, 53, 54], dtype=torch.uint8)
        expect_splits = torch.tensor([0, 1, 3, 6])
        torch.testing.assert_close(values, expect_values)
        torch.testing.assert_close(splits, expect_splits)

    def test_ragged_char_to_string(self):
        values = torch.tensor([48, 49, 50, 51], dtype=torch.uint8)
        splits = torch.tensor([0, 2, 4], dtype=torch.int32)
        arr = ragged_char_to_string(RaggedTensor(values=values, offsets=[splits]))
        expect_arr = np.array([b"01", b"23"], dtype="|S2")
        self.assertTrue(np.all(np.char.equal(arr, expect_arr)))

    def test_load_by_batch(self):
        sampler = LocalDataSampler(
            sample_tag="sample_tag",
            dedup_tag="dedup_tag",
            weight_tag="weight_tag",
            skey_name="skey_name",
        )
        batch, _ = self.mock_batch()
        sampler.load_by_batch(batch)

    def test_sample_ids(self):
        sampler = LocalDataSampler(
            sample_tag="sample_tag",
            dedup_tag="dedup_tag",
            weight_tag="weight_tag",
            skey_name="skey_name",
            put_back=False,
        )
        batch, batch_size = self.mock_batch()
        sampler.load_by_batch(batch)
        sample_tag_tensors = self.mock_sample_tag_tensors()
        dedup_tag_tensors = self.mock_dedup_tag_tensors()
        sample_cnt = 2
        sample_ids = sampler.sample_ids(
            sample_tag_tensors, dedup_tag_tensors, sample_cnt
        )
        # print(sample_ids)
        self.assertTrue(0 not in sample_ids[:sample_cnt])
        self.assertTrue(1 not in sample_ids[sample_cnt:])

    def test_sample_ids_with_avoid_conflict_with_all_dedup_tags(self):
        sampler = LocalDataSampler(
            sample_tag="sample_tag",
            dedup_tag="dedup_tag",
            weight_tag="weight_tag",
            skey_name="skey_name",
            put_back=False,
        )
        batch, batch_size = self.mock_batch()
        sampler.load_by_batch(batch)
        sample_tag_tensors = self.mock_sample_tag_tensors()
        dedup_tag_tensors = self.mock_dedup_tag_tensors()
        sample_cnt = 2
        sample_ids = sampler.sample_ids(
            sample_tag_tensors,
            dedup_tag_tensors,
            sample_cnt,
            avoid_conflict_with_all_dedup_tags=True,
        )
        print(sample_ids)
        self.assertTrue(0 not in sample_ids)
        self.assertTrue(1 not in sample_ids)

    def test_valid_sample_ids(self):
        sampler = LocalDataSampler(
            sample_tag="sample_tag",
            dedup_tag="dedup_tag",
            weight_tag="weight_tag",
            skey_name="skey_name",
            put_back=True,
        )
        batch, batch_size = self.mock_batch()
        sampler.load_by_batch(batch)

        # The sampler will be used in multi-thread, so we test it
        # in multi-thread.
        def func(i):
            dedup_tag_tensors = RaggedTensor(
                values=torch.tensor([i] * 2), offsets=[torch.tensor([0, 1, 2])]
            )
            sample_tag_tensors = self.mock_sample_tag_tensors()
            sample_cnt = 2
            neg_sample_ids = sampler.sample_ids(
                sample_tag_tensors, dedup_tag_tensors, sample_cnt
            )
            local_data_sample_ids = torch.reshape(
                torch.concat(
                    [
                        torch.reshape(dedup_tag_tensors.values(), [-1, 1]),
                        torch.reshape(neg_sample_ids, [-1, sample_cnt]),
                    ],
                    axis=1,
                ),
                [-1],
            )
            valid_sample_ids = sampler.valid_sample_ids(
                local_data_sample_ids, 5533571732986600803
            )
            # print(valid_sample_ids)
            self.assertTrue(i not in valid_sample_ids[1 : sample_cnt + 1])
            self.assertTrue(i not in valid_sample_ids[sample_cnt + 2 :])

        num_threads = 16
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=func, args=[i])
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    def test_pack_feature(self):
        sampler = LocalDataSampler(
            sample_tag="sample_tag",
            dedup_tag="dedup_tag",
            weight_tag="weight_tag",
            skey_name="skey_name",
            put_back=True,
        )
        batch, batch_size = self.mock_batch()
        sampler.load_by_batch(batch)

        # The sampler will be used in multi-thread, so we test it
        # in multi-thread.
        def func(i):
            dedup_tag_tensors = RaggedTensor(
                values=torch.tensor([i] * 2), offsets=[torch.tensor([0, 1, 2])]
            )
            sample_tag_tensors = self.mock_sample_tag_tensors()
            sample_cnt = 2
            neg_sample_ids = sampler.sample_ids(
                sample_tag_tensors, dedup_tag_tensors, sample_cnt
            )
            local_data_sample_ids = torch.reshape(
                torch.concat(
                    [
                        torch.reshape(dedup_tag_tensors.values(), [-1, 1]),
                        torch.reshape(neg_sample_ids, [-1, sample_cnt]),
                    ],
                    axis=1,
                ),
                [-1],
            )
            valid_sample_ids = sampler.valid_sample_ids(
                local_data_sample_ids, 5533571732986600803
            )
            neg_features = sampler.pack_feature(
                valid_sample_ids,
                decorate_skey=False,
                default_value=5533571732986600803,
            )
            data = neg_features["data"].values()
            dedup_tag = neg_features["dedup_tag"].values()
            torch.testing.assert_close(data, dedup_tag)

        num_threads = 8
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=func, args=[i])
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()


if __name__ == "__main__":
    unittest.main()
