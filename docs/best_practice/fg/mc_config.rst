mc配置
========

RecIS支持两种mc配置方式。MC配置表示含义为block级别的特征聚合（concat）

List模式示例
--------------------------------

.. code-block:: json

    {
        "block_name": [
            "fea1",
            "fea2",
            "fea3"
        ],
        "seq_name": [
            "sfea1",
            "sfea2"
        ]
    }

**非序列特征**

将fea1 fea2 fea3的embedding结果聚合为block_name

**序列特征**

将 seq_name_sfea1 seq_name_sfea2 的embedding结果聚合为seq_name


Dict模式示例
--------------------------------

.. code-block:: json

    {
        "block_name": [
            {"column_name": "fea1"},
            {"column_name": "fea2"},
            {"column_name": "fea3"}
        ],
        "seq_name": [
            {"column_name": "sfea1"},
            {"column_name": "sfea2"}
        ]
    }

**非序列特征**

将fea1 fea2 fea3的embedding结果聚合为block_name

**序列特征**

将 seq_name_sfea1 seq_name_sfea2 的embedding结果聚合为seq_name
