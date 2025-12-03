保存部分Sparse参数
=====================

详细API文档: :class:`recis.framework.trainer.TrainingArguments` - TrainingArguments

通过名字指定
------------------------

.. code-block:: python

    train_arg = TrainingArguments(
        # item表中的全部字段都不保存
        params_not_save=[
            "item@id",
            "item@emb",
            "item@sparse_adamw_tf_exp_avg",
            "item@sparse_adamw_tf_exp_avg_sq"],
        # ... 其他配置
    )
    # 定义trainer
    # trainer = Trainer(train_arg, ...)

自定义过滤函数
------------------------

.. code-block:: python

    # 过滤hashtable中以item开头的表
    def filter_fn(blocks):
        out_blocks = []
        for block in blocks:
            if not block.tensor_name().startswith("item"):
                out_blocks.append(block)
        return out_blocks

    train_arg = TrainingArguments(
        # item表中的全部字段都不保存
        save_filter_fn=filter_fn,
        # ... 其他配置
    )
    # 定义trainer
    # trainer = Trainer(train_arg, ...)