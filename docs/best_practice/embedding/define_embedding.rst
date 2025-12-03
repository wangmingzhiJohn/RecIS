Embedding定义
=====================

常见定义
------------

详细API文档: :class:`recis.nn.EmbeddingOption` - EmbeddingOption

最简单定义 
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # embedding_dim 维度
    # shared_name   查找的emb表名
    # combiner      聚合方式（mean/sum）

    emb_opt = EmbeddingOption(
        embedding_dim=embedding_dim,
        shared_name=shared_name,
        combiner=combiner,
        device=torch.device("cuda"),
    )

设定初始化器
~~~~~~~~~~~~~~~~~~

详细API文档: :doc:`../../api/nn/initializer`

.. code-block:: python

    # embedding_dim 维度
    # shared_name   查找的emb表名
    # combiner      聚合方式（mean/sum）

    initializer = UniformInitializer(a=-2e-5, b=2e-5)

    emb_opt = EmbeddingOption(
        embedding_dim=embedding_dim,
        shared_name=shared_name,
        combiner=combiner,
        device=torch.device("cuda"),
        initializer=initializer,
    )

设定device
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # embedding_dim 维度
    # shared_name   查找的emb表名
    # combiner      聚合方式（mean/sum）

    device = torch.device("cuda")   # GPU
    # device = torch.device("cuda")   # CPU

    emb_opt = EmbeddingOption(
        embedding_dim=embedding_dim,
        shared_name=shared_name,
        combiner=combiner,
        device=device,
    )

不参与训练模式
~~~~~~~~~~~~~~~~~~

**不允许新ID插入，且不更新**

.. code-block:: python

    # embedding_dim 维度
    # shared_name   查找的emb表名
    # combiner      聚合方式（mean/sum）

    emb_opt = EmbeddingOption(
        embedding_dim=embedding_dim,
        shared_name=shared_name,
        combiner=combiner,
        device=torch.device("cuda"),
        admit_hook=AdmitHook("ReadOnly"),   # 禁止新ID插入
        trainable=False,                    # 不参与反向更新
    )

**不允许新ID插入(已存在ID更新)**

.. code-block:: python

    # embedding_dim 维度
    # shared_name   查找的emb表名
    # combiner      聚合方式（mean/sum）

    emb_opt = EmbeddingOption(
        embedding_dim=embedding_dim,
        shared_name=shared_name,
        combiner=combiner,
        device=torch.device("cuda"),
        admit_hook=AdmitHook("ReadOnly"),   # 禁止新ID插入
    )

**不允许新ID插入(已存在ID更新)**

.. code-block:: python

    # embedding_dim 维度
    # shared_name   查找的emb表名
    # combiner      聚合方式（mean/sum）

    emb_opt = EmbeddingOption(
        embedding_dim=embedding_dim,
        shared_name=shared_name,
        combiner=combiner,
        device=torch.device("cuda"),
        admit_hook=AdmitHook("ReadOnly"),   # 禁止新ID插入
    )

设定dtype
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # embedding_dim 维度
    # shared_name   查找的emb表名
    # combiner      聚合方式（mean/sum）
    # dtype         emb参数类型

    emb_opt = EmbeddingOption(
        embedding_dim=embedding_dim,
        shared_name=shared_name,
        combiner=combiner,
        device=torch.device("cuda"),
        dtype=dtype,
    )

特征准入与特征淘汰
~~~~~~~~~~~~~~~~~~

:doc:`filter`