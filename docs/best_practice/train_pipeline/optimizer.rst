优化器
================

稀疏优化器
----------------

详细API文档： :doc:`../../api/optim`

典型使用
~~~~~~~~~

.. code-block:: python

    from recis.nn.modules.hashtable import filter_out_sparse_param
    
    # 过滤模型中的hashtable参数
    sparse_params = filter_out_sparse_param(model)
    sparse_optimizer = SparseAdamW(sparse_params, lr=0.001)


稠密优化器
--------------------------------

torch原生
~~~~~~~~~~~~~~~~~~

详细API文档：https://docs.pytorch.org/docs/stable/optim.html

NamedOptimizer
~~~~~~~~~~~~~~~~~~

TODO(lanling.ljw)

典型使用
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # 原生稠密优化器
    from torch.optim import AdamW
    dense_opt = AdamW(params=model.parameters(), lr=dense_lr, weight_decay=1e-6)

    # NamedOptimizer
    from recis.optim.named_optimizer import NamedAdamW
    dense_opt = NamedAdamW(params=model.named_parameters(), lr=dense_lr, weight_decay=1e-6)
