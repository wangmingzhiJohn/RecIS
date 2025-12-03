特征准入与特征过滤
=====================

详细API文档: :doc:`../../api/nn/filter`

特征准入
--------

Embedding定义
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
        admit_hook=AdmitHook("ReadOnly"),   # 禁止新ID插入
    )

特征淘汰
--------

Embedding定义
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
        # 淘汰2000个step中没有出现的ID
        filter_hook=FilterHook("GlobalStepFilter", {"filter_step": 2000}),
    )

训练Hook定义
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # trainer = ... 定义训练器

    # 每1000个step，开启一次过滤操作
    filter_hook = HashTableFilterHook(1000)
    trainer.add_hooks([filter_hook])