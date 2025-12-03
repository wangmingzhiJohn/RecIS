样本转换
===================

RecIS 支持在IO模块进行数据预处理。通常用来生成特征，或特征格式转换。

非结构化压缩特征处理
-----------------------------

.. code-block:: python

    # 取出结果中，非结构化压缩的列（非结构化压缩必要行为）
    transform_fn = lambda x: x[0]

    dataset = OdpsDataset(
        # ..., 其他参数定义
        transform_fn=transform_fn, # 传入数据转换方法
    )

嵌套转换与特征拷贝
-----------------------------

.. code-block:: python

    def make_copy_features(copy_map):
        def copy_features(x):
            for src, dst in copy_map.items():
                if feat.get(src, None) is not None:
                    x[feat[dst]] = x[feat[src]].clone()
            return x
        return copy_features

    # copy_map 拷贝配置
    # 嵌套的转换特征，先取出非结构化压缩特征，然后进行特征拷贝
    transform_fn = [lambda x: x[0], make_copy_features(copy_map)]

    dataset = OdpsDataset(
        # ..., 其他参数定义
        transform_fn=transform_fn, # 传入数据转换方法
    )

注意：不要通过该方法做Hash Bucketize等操作，这些操作在Feature Engine中定义才会享受到优化加速。
