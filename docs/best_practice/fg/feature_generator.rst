FG模块
================

详细API文档: :doc:`../../api/fg`

创建FG
--------

**标准用法**

.. code-block:: python

    fg = build_fg(
        fg_conf_path,
        mc_conf_path=mc_conf_path,
    )
    # 单独添加label字段
    fg.add_label("label")
    # 单独添加sample id字段
    fg.add_id("id")

**IO中做Hash操作**

某些场景下，需要将Hash处理放到IO中进行，需加上hash_in_io=True参数

.. code-block:: python

    fg = build_fg(
        fg_conf_path,
        mc_conf_path=mc_conf_path,
        hash_in_io=True,
    )
    # 单独添加label字段
    fg.add_label("label")
    # 单独添加sample id字段
    fg.add_id("id")

**原始数据中已经做过Hash处理**

某些场景下，训练数据已经将string类型特征在数据源处理阶段已经做过hash（没有hash bucketize）时，需加上already_hashed=True参数

.. code-block:: python

    fg = build_fg(
        fg_conf_path,
        mc_conf_path=mc_conf_path,
        already_hashed=True,
    )
    # 单独添加label字段
    fg.add_label("label")
    # 单独添加sample id字段
    fg.add_id("id")

为Dataset添加特征
------------------------

.. code-block:: python

    # dataset = ... 定义dataset
    # fg = ... 定义FG
    # 无需手动调用varlen_feature/fixedlen_feature
    # 直接通过fg添加特征
    fg.add_io_features(dataset)

.. _fg_model:

创建Sparse Model
------------------------

.. code-block:: python

    from recis.nn.modules.models import RecISModel
    
    class MyModel(nn.Module):
        def __init__(self, fg: FG):
            # 创建 sparse model，包括FeatureEngine和EmbeddingEngine
            self.sparse_arch = RecISModel.from_fg(fg)
            # ...

        def forward(self, samples):
            # 输入为dataset输出结果，key为特征名，value为tensor的dict
            samples, ids, labels = self.sparse_arch(samples)
            """
            3个输出
            1. samples: 经过特征转换和特征查找后，根据mc concat聚合之后的block embedding
            2. ids: dict，key为调用add_id时的名字，value为对应的值
            3. labels: dict, key为调用add_label时的名字，value为对应的值
            """
            # ...

    # fg = ... 定义FG
    # model = MyModel(fg)