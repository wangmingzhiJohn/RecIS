特征处理(Feature Engine)
=============================

详细API文档: :class:`recis.features.FeatureEngine`

创建 Feature Engine
----------------------

.. code-block:: python

    # input_data: dataset读取的数据集
    features = []
    # 添加特征user_id
    features.append(
        Feature("user_id").add_op(SelectField("user_id")).add_op(Mod(10000))
    )
    # 添加其他特征
    # features.append(...)

    # 创建 feature engine
    feature_engine = FeatureEngine(features)
    # 获取特征转换结果
    output_data = feature_engine(input_data)