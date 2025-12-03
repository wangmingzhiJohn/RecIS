数据处理流程
=================

.. toctree::
   :maxdepth: 2
   
   dataset
   window_io
   pre_transform/index

简单示例
----------

.. code-block:: python

   def get_odps_dataset():
       worker_idx = int(os.environ.get("RANK", 0))
       worker_num = int(os.environ.get("WORLD_SIZE", 1))
       transform_fn = [lambda x: x[0]]
       # 定义dataset
       dataset = OdpsDataset(
         batch_size=1024,
         worker_idx=worker_idx,
         worker_num=worker_num,
         read_threads_num=2,
         prefetch=1,
         is_compressed=False,
         drop_remainder=True,
         transform_fn=transform_fn,
         dtype=torch.float32,
         device="cuda",
         save_interval=100,
       )
       # 添加path
       dataset.add_path(
         "odps://xxx/tables/xxx/ds=xxx",
       )
       # 添加特征
       dataset.varlen_feature(
         name="item_id",
         trans_int8=True,
       )
       dataset.varlen_feature(
         name="user_id",
         trans_int8=True,
       )
       dataset.fixedlen_feature(
         name="rate",
         default_value=[0.0]
       )
       dataset.fixedlen_feature(
         name="label",
         default_value=[0.0]
       )
       return dataset
