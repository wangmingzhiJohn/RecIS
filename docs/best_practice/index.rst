最佳实践
=================

如果您刚接触RecIS，那么快速开始部分可以为您讲解如何从0开始构建一个RecIS模型，您可以参阅各个部分的 **简单示例** 中的代码，快速组织出一个Demo。
跳转快速开始：:ref:`best_practice-quick_start`

如果您已经有了一个RecIS模型，正在基础模型上迭代，您可以分别参考下面文档中各部分内容，我们已经将常见使用方式通过示例的形式展现出来。

如果您是一个资深用户，下面的文档都无法满足您的需求，那么可以参考API部分的文档，查看更详细的API介绍。
跳转API文档：:doc:`../api/index`

.. toctree::
   :maxdepth: 2
   :caption: 目录结构:

   data_pipeline/index
   feature/index
   embedding/index
   fg/index
   model/index
   train_pipeline/index
   checkpoint/index
   rtp/index

.. _best_practice-quick_start:

快速开始
------------

使用RecIS构建一个推荐模型需要在代码中完成 **4个定义** 与 **1个流程**

**4个定义** 分别为：

1. Dataset定义：:doc:`data_pipeline/index`

2. 特征转换定义：:doc:`feature/index`

3. Embedding定义：:doc:`embedding/index`

4. 模型定义：:doc:`model/index`

RecIS的FG模块可以帮助简化上述 **4个定义** 流程

- FG模块：:doc:`fg/index`

**1个流程** 为：

- 训练流程pipeline创建：:doc:`train_pipeline/index`

简单示例
------------

**基本使用**
            
.. code-block:: python

    # 参考Dataset定义
    train_dataset = get_odps_dataset()

    # 参考特征转换定义
    feature_conf = get_feature_conf()

    # 参考Embedding定义
    emb_conf = get_emb_conf()

    # 参考模型定义
    graph = MyModel(feature_conf, emb_conf)
    graph = graph.cuda()

    # 参考训练流程pipeline创建
    dense_opt, sparse_opt = get_optimizer(graph)
    trainer = get_trainer(dataset, graph, dense_opt, sparse_opt)
    # 开始训练
    trainer.train()


**使用FG模块**

.. code-block:: python

    # 参考FG模块
    fg = get_fg()
    train_dataset = get_odps_dataset_by_fg(fg)

    # 参考模型定义
    graph = MyModel(fg)
    graph = graph.cuda()

    # 参考训练流程pipeline创建
    dense_opt, sparse_opt = get_optimizer(graph)
    trainer = get_trainer(dataset, graph, dense_opt, sparse_opt)
    # 开始训练
    trainer.train()


分布式相关
------------

- 提交分布式任务：:ref:`nebula`
- 分布式完整Demo：:ref:`demo`
- 常见问题：:ref:`qa`

其他工具
------------

- 指标监控：:ref:`metrics`
- Tensorflow迁移Recis：:ref:`tf2recis`
- XrecV1版本迁移Recis：:ref:`xrec2recis`
- ODPS数据导出为Recis：:ref:`odps2recis`
