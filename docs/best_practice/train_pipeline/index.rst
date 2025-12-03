运行Pipeline
=================

.. toctree::
   :maxdepth: 2
   
   optimizer
   pipeline
   hooks

简单示例
----------

.. code-block:: python

    def get_optimizer(model):
        sparse_param = filter_out_sparse_param(model)
        dense_opt = AdamW(params=model.parameters(), lr=0.001, weight_decay=1e-6)
        sparse_opt = SparseAdamW(sparse_param, lr=0.001, weight_decay=1e-6)
        return dense_opt, sparse_opt

    def get_trainer(dataset, model, dense_opt, sparse_opt):
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
        return trainer
