Feature Generator(FG模块)
===================================

.. toctree::
   :maxdepth: 2
   
   feature_generator
   fg_config
   mc_config

RecIS提供一套标准的 fg / mc 解析处理模块，能够自动构建模型的稀疏部分，包括：

1.  IO：根据配置，为io添加需要消费的特征列
2. FeatureEngine：根据配置，生成特征转换方式，创建feature engine
3. EmbeddingEngine：根据配置，生成embedding option，并创建embedding engine

RecIS的FG模块集成了 广告rtp fg 和 主搜rtp fg 的解析处理逻辑，并在rtp在线配置基础上做了一定拓展，为训练使用。
使用RecIS的FG模块需要准备2个配置文件：

1. fg.json：特征处理以及查表配置
2. mc.json：block级别的特征聚合（concat）配置

简单示例
-----------

.. code-block:: python

    def get_fg():
        fg = build_fg(
            "./conf/fg.json", # fg配置文件路径
            mc_conf_path="./conf/mc.json", # mc配置文件路径
        )
        fg.add_label("label")
        return fg

   def get_odps_dataset_by_fg(fg):
       worker_idx = int(os.environ.get("RANK", 0))
       worker_num = int(os.environ.get("WORLD_SIZE", 1))
       transform_fn = [lambda x: x[0]]
       # 定义dataset
       dataset = OdpsDataset(
         batch_size=1024,
         worker_idx=worker_idx,
         worker_num=worker_num,
         read_threads_num=2,
         prefetch=1,
         is_compressed=False,
         drop_remainder=True,
         transform_fn=transform_fn,
         dtype=torch.float32,
         device="cuda",
         save_interval=100,
       )
       # 添加path
       dataset.add_path(
         "odps://xxx/tables/xxx/ds=xxx",
       )
       fg.add_io_features(dataset)
       return dataset


Tips
-----------

1. fg中所需要的字段一定要正确，否则处理可能不符合预期
2. 确保mc中出现的列和特征名，一定会被模型用到，请手动删除不需要的列配置

参考文档
-----------

参考：:ref:`rtp`
