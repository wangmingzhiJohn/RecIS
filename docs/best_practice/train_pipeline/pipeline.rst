训练流程
================

详细API文档： :doc:`../../api/framework`

典型使用
----------------

训练
~~~~~~~~~~~~

.. code-block:: python

    # dataset: 训练数据集
    # model: 模型
    # dense_opt: 稠密参数优化器
    # sparse_opt: 稀疏参数优化器
    
    train_config = TrainingArguments(
        output_dir="./ckpt/",
        model_bank=None,
        log_steps=10,
        save_steps=1000,
    )
    trainer = Trainer(
        model=model,
        args=train_config,
        train_dataset=dataset,
        dense_optimizers=(dense_opt, None),
        sparse_optimizer=sparse_opt,
    )
    # 开始训练
    trainer.train()

预测
~~~~~~~~~~~~

.. code-block:: python

    # dataset: evaluate数据集
    # model: 模型
    # model_bank_conf: 预测任务需要加载的模型配置
    
    train_config = TrainingArguments(
        output_dir=None,
        model_bank=model_bank_conf,
        log_steps=10,
        save_steps=1000,
    )
    trainer = Trainer(
        model=model,
        args=train_config,
        eval_dataset=dataset,
    )
    # 开始训练
    trainer.evaluate()

边训练边预测
~~~~~~~~~~~~

.. code-block:: python

    # train_dataset: 训练数据集
    # eval_dataset: evaluate数据集
    # model: 模型
    # dense_opt: 稠密参数优化器
    # sparse_opt: 稀疏参数优化器
    
    train_config = TrainingArguments(
        output_dir="./ckpt/",
        model_bank=None,
        log_steps=10,
        save_steps=1000,
    )
    trainer = Trainer(
        model=model,
        args=train_config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        dense_optimizers=(dense_opt, None),
        sparse_optimizer=sparse_opt,
    )
    # 开始训练
    trainer.train_and_evaluate()

导图
----------------

参考: :doc:`../rtp/index`