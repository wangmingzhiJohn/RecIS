主搜-原生交付
========================

详细API文档: :class:`recis.framework.Exporter`

使用FG RecISModel模块交付流程（推荐）
------------------------------------------------------------

RecISModel 以及 FG 模块已经为导图处理做过适配，使用该模块的模型可以轻松完成在线交付。
FG API实践：:doc:`../fg/index`

fg规范
~~~~~~~~~~~~~~~~~~

参考：:ref:`rtp-guifan`

tips：

1. 主搜rtp不支持hash_bucket_size为0的fg定义

2. 主搜rtp不支持multihash的fg特征定义

3. 能被RecIS FG模块解析的fg即可满足在线规范

模型定义规范
~~~~~~~~~~~~~~~~~~

tips：

1. 交付主搜rtp的模型在使用RecISModel.from_fg方法时，不能将split_seq设置为True

2. dense arch的输出为Tensor 或 Tuple[Tensor] 表示需要在线打分结果

.. code-block:: python

    class MyModel(nn.module):
        def __init__(self, fg, **kwargs):
            super().__init__()
            # sparse模型定义，from_fg方法不支持 split_seq 参数为True
            self.sparse_arch = RecISModel.from_fg(fg)
            # dense模型定义，输入为RecISModel的返回结果，输出为需要在线打分的tensor
            self.dense_arch = MyDense(model_conf, fg)

        def forward(self, samples):
            samples, sample_ids, labels = self.sparse_arch(samples)
            predicts = self.dense_arch(samples)
            loss = xxx
            return loss

使用Exporter导出在线dense图
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from framework.exporter import Exporter
    # fg = xxx
    # dataset = xxx
    # model = MyModel(fg)
    exporter = Exporter(
        model=model,
        # sparse部分模型在model中的名字
        sparse_model_name="sparse_arch",
        # dense部分模型在model中的名字
        dense_model_name="dense_arch",
        # 输入数据
        dataset=dataset,
        # 需要上线的ckpt路径或Mos uri（精确到版本）
        ckpt_dir=get_config("export", "output_dir"),
        # 在线文件导出路径
        export_dir=get_config("export", "output_dir"),
        # dense optimizer
        dense_optimizer=None,
        # dense部分导出的文件夹名，和在线配置相关
        export_folder_name="fx_user_model",
        # dense部分导出的模型名，和在线配置相关
        export_model_name=get_config("export", "model_name"),
        # dense arch输出结果（在线打分结果）名
        export_outputs=get_config("export", "out_node_names"),
        # 任务feature generator
        fg=fg,
        # 是否需要将sparse参数过滤optimizer信息进行交付，默认为False
        filter_sparse_opt=False,
    )
    exporter.export()

- model：完整模型
- sparse_model_name：model中sparse部分名称，一般为RecISModel
- dense_model_name：model中dense部分名称，输入为RecISModel的输出结果，输出为Tensor 或 Tuple[Tensor]表示打分结果
- dataset：输入的io数据
- ckpt_dir：需要在线交付的模型地址，精确到模型版本
- export_dir：导出的在线模型文件地址
- dense_optimizer：如果模型完成过反向计算，需传入dense optimizer，用于清理grad状态
- export_folder_name：dense部分在线文件夹名，默认为"fx_user_model"，需要和在线配置一致
- export_model_name：在线模型名，默认为"user_model"，需要和在线配置一致
- export_outputs：dense model输出tensor对应的在线打分名称
- fg：feature_generator
- filter_sparse_opt：是否将sparse部分参数过滤optimizer相关信息后导出到export_dir中。默认为False。

自定义模型交付流程（不推荐）
------------------------------------------------------------

参考：:ref:`rtp-jiaofu`

其他RTP相关文档
----------------------------------

参考：:ref:`rtp`
