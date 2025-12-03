特征定义
=============================

详细API文档: :class:`recis.features.feature.Feature`

常见特征定义
---------------

输入原始String，直接Hash处理
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

原始输入的特征数据类型为String；且没有在dataset中做hash，并开启 trans_int8 = True.

.. code-block:: python

    # fea_in_name 特征输入名
    # fea_out_name 特征输出名
    # hash_type hash类型
    # hash_bucket_size 特征hash桶大小

    fea_conf = Feature(fea_out_name).add_op(
        SelectField(fea_in_name)
    )
    fea_conf = fea_conf.add_op(Hash(hash_type))
    if hash_bucket_size > 0:
        fea_conf = fea_conf.add_op(Mod(hash_bucket_size))

ID特征取模
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

原始输入的特征数据类型为Int
或原始输入的特征数据类型为String，且已经在dataset中做hash。

.. code-block:: python

    # fea_in_name 特征输入名
    # fea_out_name 特征输出名
    # hash_bucket_size 特征hash桶大小

    fea_conf = Feature(fea_out_name).add_op(
        SelectField(fea_in_name)
    )
    if hash_bucket_size > 0:
        fea_conf = fea_conf.add_op(Mod(hash_bucket_size))

数值特征分桶
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

原始输入的特征数据类型为Float

.. code-block:: python

    # fea_in_name 特征输入名
    # fea_out_name 特征输出名
    # dim 特征输入维度
    # boundaries 特征桶边界值

    fea_conf = Feature(fea_out_name).add_op(
        SelectField(fea_in_name, dim=dim)
    )
    fea_conf = fea_conf.add_op(Bucketize(boundaries))

MultiHash
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**String类型特征的Multihash**

原始输入的特征数据类型为String；且没有在dataset中做hash，并开启 trans_int8 = True.

.. code-block:: python

    # fea_in_name 特征输入名
    # fea_out_name 特征输出名
    # hash_type hash类型
    # hash_bucket_size 特征hash桶大小
    # num_buckets multihash的4次hash分桶数
    # prefix multihash的4次hash前缀

    fea_conf = Feature(fea_out_name).add_op(
        SelectField(fea_in_name)
    )
    fea_conf = fea_conf.add_op(Hash(hash_type))
    if hash_bucket_size > 0:
        fea_conf = fea_conf.add_op(Mod(hash_bucket_size))
    fea_conf = fea_conf.add_op(IDMultiHash(num_buckets, prefix))

**ID类型特征的Multihash**

原始输入的特征数据类型为Int
或原始输入的特征数据类型为String，且已经在dataset中做hash。

.. code-block:: python

    # fea_in_name 特征输入名
    # fea_out_name 特征输出名
    # hash_bucket_size 特征hash桶大小
    # num_buckets multihash的4次hash分桶数
    # prefix multihash的4次hash前缀

    fea_conf = Feature(fea_out_name).add_op(
        SelectField(fea_in_name)
    )
    if hash_bucket_size > 0:
        fea_conf = fea_conf.add_op(Mod(hash_bucket_size))
    fea_conf = fea_conf.add_op(IDMultiHash(num_buckets, prefix))

序列截断
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # seq_length 序列长度
    # dtype 序列截断前数据类型

    fea_conf = fea_conf.add_op(
        SequenceTruncate(
            seq_len=seq_length,
            truncate=True,
            truncate_side="right",
            check_length=False,
            n_dims=3,
            dtype=dtype,
        )
    )

