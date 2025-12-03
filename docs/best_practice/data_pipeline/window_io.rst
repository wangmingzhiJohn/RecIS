Window IO 窗口训练
===================

详细API文档: :ref:`window_io` - WindowIO

ODPS 平均切分为N个窗口
-----------------------------

.. code-block:: python

    dataset_class = make_odps_window_io(split_num=4) # 分成4个窗口

    """ 定义dataset
    worker_idx = int(os.environ.get("RANK", 0))
    worker_num = int(os.environ.get("WORLD_SIZE", 1))
    dataset = dataset_class(...)
    """

ODPS 按照数据条数切分窗口
-----------------------------

.. code-block:: python

    dataset_class = make_odps_window_io(row_num=10000) # 每个窗口10000条样本

    """ 定义dataset
    worker_idx = int(os.environ.get("RANK", 0))
    worker_num = int(os.environ.get("WORLD_SIZE", 1))
    dataset = dataset_class(...)
    """

Lake 按照时间间隔切分窗口
-----------------------------

.. code-block:: python

    dataset_class = make_lake_stream_window_io(step_mins=60) # 60min一个窗口

    """ 定义dataset
    worker_idx = int(os.environ.get("RANK", 0))
    worker_num = int(os.environ.get("WORLD_SIZE", 1))
    dataset = dataset_class(...)
    """

注意
-------------------

window io模式下 save_interval 参数必须为 None。