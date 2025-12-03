创建数据集
===============

创建ODPS数据集
--------------

详细API文档: :class:`recis.io.OdpsDataset` - ODPS Dataset

.. code-block:: python

    worker_idx = int(os.environ.get("RANK", 0))
    worker_num = int(os.environ.get("WORLD_SIZE", 1))
    transform_fn = [lambda x: x[0]]
    dataset = OdpsDataset(
        batch_size=1024,        # Batch size
        worker_idx=worker_idx,  # 当前rank
        worker_num=worker_num,  # 总rank数
        read_threads_num=2,     # 读取数据线程数
        prefetch=1,             # 数据预取个数
        is_compressed=False,    # 是否为结构化压缩表
        drop_remainder=True,    # 丢弃最后一组不满bs的数据
        transform_fn=transform_fn,
        dtype=torch.float32,    # 浮点数类型数据输出dtype
        device="cuda",          # batch数据直接place到cuda上
        save_interval=100,      # IO状态保存间隔
    )

创建Lake数据集
--------------

详细API文档: :class:`recis.io.LakeStreamDataset` - Lake Dataset

.. code-block:: python

    worker_idx = int(os.environ.get("RANK", 0))
    worker_num = int(os.environ.get("WORLD_SIZE", 1))
    transform_fn = [lambda x: x[0]]
    dataset = LakeStreamDataset(
        batch_size=1024,        # Batch size
        worker_idx=worker_idx,  # 当前rank
        worker_num=worker_num,  # 总rank数
        read_threads_num=2,     # 读取数据线程数
        prefetch=1,             # 数据预取个数
        is_compressed=False,    # 是否为结构化压缩表
        drop_remainder=True,    # 丢弃最后一组不满bs的数据
        dtype=torch.float32,    # 浮点数类型数据输出dtype
        transform_fn=transform_fn,
        device="cuda",          # batch数据直接place到cuda上
        save_interval=100,      # IO状态保存间隔
    )

添加特征
--------------

数值特征(非序列特征)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    dataset.fixedlen_feature(
        name=fn,                    # 特征列名
        default_value=[0.0] * dim   # 默认值
    )

数值特征(序列特征)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    dataset.varlen_feature(
        name=fn                     # 特征列名
    )

稀疏ID特征
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    dataset.varlen_feature(
        name=fn,                    # 特征列名
    )

稀疏String特征
~~~~~~~~~~~~~~~~~~~~~~

**IO读取过程中做string哈希**

.. code-block:: python

    dataset.varlen_feature(
        name=fn,                        # 特征列名
        hash_type="farm",               # 哈希类型
        hash_bucket=hash_bucket_size    # 哈希桶大小
    )

**后续Feature转换过程中做string哈希**

.. code-block:: python

    dataset.varlen_feature(
        name=fn,                        # 特征列名
        trans_int8=True,                # 将string类型读取成int8类型
    )

此时 trans_int8 参数必须设置为True，否则后续Feature转换过程无法正确处理。

特殊参数
--------------

**save_interval**

IO状态保存间隔，IO状态可以保证分布式任务发生failover后，任务能够从上次IO状态继续。因此IO状态保存间隔越小，在发生failover时丢失的数据越少；但性能可能受损。

**device**

batch数据直接place的device，支持"cpu" "pin" "cuda"。"cpu" "pin" 时，后续trainer需要设定 data_to_cuda 为True，才能正确训练，详见 class:`recis.framework.Trainer`

**prefetch**

数据处理流程预取个数，通常1即可