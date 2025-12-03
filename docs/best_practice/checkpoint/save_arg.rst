保存参数设置
========================

详细API文档: :class:`recis.framework.trainer.TrainingArguments` - TrainingArguments

每 N step 保存一次(结束额外保存)
-------------------------------------

.. code-block:: python

    train_arg = TrainingArguments(
        save_steps=1000, # 1000step保存一次模型
        # ... 其他配置
    )
    # 定义trainer
    # trainer = Trainer(train_arg, ...)

每 window 保存一次(结束不额外保存)
-------------------------------------

.. code-block:: python

    train_arg = TrainingArguments(
        save_steps=None,
        save_every_n_windows=1,
        save_end=False,
        # ... 其他配置
    )
    # 定义trainer
    # trainer = Trainer(train_arg, ...)

每 epoch 保存一次(结束不额外保存)
-------------------------------------

.. code-block:: python

    train_arg = TrainingArguments(
        save_steps=None,
        save_every_n_windows=None,
        save_every_n_epochs=1,
        save_end=False,
        # ... 其他配置
    )
    # 定义trainer
    # trainer = Trainer(train_arg, ...)

某些Sparse参数不保存
-------------------------------

:doc:`save_part`