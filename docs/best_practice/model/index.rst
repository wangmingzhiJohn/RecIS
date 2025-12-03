模型定义
====================

稀疏模型
----------

稀疏部分的定义由特征转换（Feature Engine）与向量查找（Embedding Engine）组建完成的，可参考对应文档：

- :doc:`../feature/engine`

- :doc:`../embedding/engine`

另外可以通过FG模块快速完成上述两部分的定义，可参考

- :ref:`fg_model`

稠密模型
----------

模型的稠密部分可以直接通过torch的原生api自由定义

简单示例
----------

使用Feature Engine / Embedding Engine构建
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class MyModel(nn.Module):
        def __init__(self, feature_conf, emb_opt):
            self.feature_engine = FeatureEngine(feature_conf)
            self.embedding_engine = EmbeddingEngine(emb_opt)
            # ...

        def forward(self, samples):
            samples = self.feature_engine(samples)
            samples = self.embedding_engine(samples)
            label = samples.pop("label")
            # ...

使用FG模块构建
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class MyModel(nn.Module):
        def __init__(self, fg: FG):
            self.sparse_arch = RecISModel.from_fg(fg)
            # ...

        def forward(self, samples):
            samples, ids, labels = self.sparse_arch(samples)
            label = labels["label"]
            # ...