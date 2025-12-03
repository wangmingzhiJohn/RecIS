Embedding处理
=====================

Embedding Engine
---------------------

详细API文档: :class:`recis.nn.EmbeddingEngine` - Embedding Engine

.. code-block:: python

    # input_data: Feature Engine转换好的特征集合
    emb_options = {}
    # 添加特征user_id
    emb_options["user_id"] = EmbeddingOption(
        embedding_dim=128,
        shared_name="user_embedding",
        combiner="sum",
        device=torch.device("cuda"),
    )
    # 添加其他特征
    # emb_options[name] = EmbeddingOption(...)

    # 创建 feature engine
    engine = EmbeddingEngine(emb_options)
    # 获取embedding结果
    output_data = engine(input_data)

注意
---------------------

目前仅支持coalesced模式，ID高12位会被抹除