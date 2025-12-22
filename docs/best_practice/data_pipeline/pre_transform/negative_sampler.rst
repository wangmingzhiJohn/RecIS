负采样（阿里内部功能）
===================

为了避免分布式采样，同时仍能将完整的负样本表加载到内存中，我们设计了一种基于本地负采样服务的解决方案。

在每个机器节点上，仅启动一个负采样服务器进程，该进程由该节点上的所有训练进程共享。在任务初始化阶段，负样本表会被加载到该服务器中。在训练过程中，每个训练进程通过向本地服务器发起RPC请求来完成负采样。

LocalRpcDataSampler 是每个训练进程的负采样客户端。该对象在初始化时会自动启动负采样服务器，所有采样任务均可通过该对象完成。

.. note::
    由于需要将完整的负样本表加载到每台物理机的内存中，建议采用整机训练模式（例如，单机8卡），以便充分利用整个系统的内存资源。
    非整机训练模式也可以支持，但是会占用较多内存，需要用户自行取舍。另外暂时不支持非整机模式下的碎卡训练。

使用样例:

.. code-block:: python

    from recis.io.odps_dataset import OdpsDataset, get_table_size
    from recis.data.local_rpc_data_sampler import LocalRpcDataSampler
    from recis.framework.trainer import Trainer


    # 获取 dataset
    def get_dataset(table, batch_size, varlen_features, worker_num, worker_idx, **kwargs):
        dataset = OdpsDataset(
            batch_size, worker_idx=worker_idx, worker_num=worker_num,
            **kwargs
        )  # 初始化数据集
        dataset.add_path(table)  # 添加 ODPS 表
        # 向数据集中添加特征
        for f in varlen_features:
            dataset.varlen_feature(f)


    WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
    RANK = int(os.getenv("RANK", 0))
    # 获取正样本数据集，在 RecIS 中使用负采样功能时不建议使用 pytorch 的 dataloader，因为 pytorch 的
    # dataloader 会启动子进程来进行并发，会容易踩坑。这里使用 RecIS 提供的 dataset，
    # 通过设置 prefetch_transform=True 的来并发执行负采样流程
    pos_dataset = get_dataset(
        pos_table, pos_batch_size, pos_varlen_features, WORLD_SIZE, RANK,
        prefetch_transform=True
    )

    # 初始化负采样器，此负采样器会自动拉起负采样服务端进程
    neg_sampler = LocalRpcDataSampler(
        # 负样本表中的类别特征名称，为单值稀疏特征（即array<bigint>类型），负样本表中必须包含此列，
        # 如果不需要类别特征，将负样本表中的此特征全置为1即可
        sample_tag,
        # 负样本表中用于去重的特征，通常为 item id，为单值稀疏特征（即array<bigint>类型），
        # 采样得到的结果基于此列和正样本进行去重
        dedup_tag,
        # 负样本表中的权重特征，为稠密特征（即 float 或 double 类型），采样时按照权重进行采样
        # 如果不需要权重，将负样本表中的此特征全置为 1.0 即可
        weight_tag,
        # 废弃字段，但是因为历史遗留问题，负样本表中仍需要包含此列，类型为字符串，样本表中全置为空串即可
        skey_name,
    )

    # 只有 local rank 为 0 的训练进程加载负样本表
    # 注意这里通过环境变量 LOCAL_PROCESS_RANK 判断 local rank，但是该环境变量在整机训练
    # 和非整机训练的两种情况下含义不一样，本样例中展示的是整机训练的例子，如果是非整机训练，需要用
    # 户在程序最开始手动设置环境变量 os.environ['LOCAL_PROCESS_RANK'] = '0'
    if int(os.environ.get("LOCAL_PROCESS_RANK", 0)) == 0:
        neg_table_size = get_table_size(
            neg_table
        )  # 获取负样本表中的样本数量
        # 获取负样本数据集
        neg_dataset = get_dataset(
            neg_table,
            neg_table_size,
            neg_varlen_features,
            # 注意：对于负样本数据集，worker_num 和 worker_idx 必须分别设置为 1 和 0。
            # 否则会变成分布式读取，导致每台机器上的负样本表不完整。
            worker_num=1,
            worker_idx=0,
        )
        neg_data = next(iter(neg_dataset))[1]  # 读取所有负样本
        neg_sampler.reload(neg_data)  # 将负样本加载到采样器中


    # 负采样函数，入参 batch 是正样本
    def neg_sampling_fn(batch):
        # 正样本中的类别特征
        sample_tag_ragged_tensor = batch[0][sample_tag]
        # 正样本中的用于采样和去重的特征，通常为 item id
        dedup_tag_ragged_tensor = batch[0][dedup_tag]
        # 为每个正样本采样 (sample_size - 1) 个负样本。
        # sample_cnts 中的值可以不相等，支持动态负采样。
        sample_cnts = sample_size - torch.ones_like(
            dedup_tag_ragged_tensor.values()
        )
        # 传入 sample_tag 和 dedup_tag，获取一组负样本的 item id
        neg_sample_ids = neg_sampler.sample_ids(
            dedup_tag_ragged_tensor=dedup_tag_ragged_tensor,
            sample_cnts=sample_cnts,
            sample_tag_ragged_tensor=sample_tag_ragged_tensor,
        )
        # 处理部分正样本可能不在负样本表中的情况，用默认值填充，这里假设默认值是 murmurhash(0)
        # 用户可以选择其他默认值，或者不执行此逻辑
        pos_ids = neg_sampler.valid_sample_ids(
            dedup_tag_ragged_tensor.values(), default_value=5533571732986600803
        )
        # 拼接正样本和负样本 ID
        # 例如，若正样本 ID 为 [1,2,3]，采样数为 2，采样的负样本 ID 为 [5,6,7,8,9,10]，
        # 则拼接结果为 [1,5,6,2,7,8,3,9,10]
        pos_neg_ids = neg_sampler.combine_vector_with_sample_counts(
            pos_ids, sample_cnts, neg_sample_ids
        )

        # 使用拼接后的正负样本 ID，从负样本表中查询所有特征，查询的 key 为 dedup_tag
        output_sample_table = neg_sampler.pack_feature(
            pos_neg_ids, default_value=5533571732986600803
        )
        # 将原始正样本与负样本合并，其中 batch 是原始正样本表中的特征，
        # output_sample_table 是正负样本 ID 拼接后从负样本表中查询出来的特征，
        # 此函数会将原始正样本 item 的特征扩展到采样出来的负样本 item
        combined_batch = neg_sampler.combine(
            batch, output_sample_table, sample_cnts
        )
        return combined_batch


    # 将负采样转换添加到正样本数据集上，上面构造 pos_dataset 时
    # 设置了 prefetch_transform=True，这样负采样流程就会和主流程并发起来
    pos_dataset.transform_ragged_batch(neg_sampling_fn)

    # 构造模型
    model = CustomModel()

    # 构造 trainer
    trainer = Trainer(
        model=model,
        train_dataset=pos_dataset,
        dense_optimizers=(dense_opt, None),
        sparse_optimizer=sparse_top,
        data_to_cuda=True)

    trainer.train()
