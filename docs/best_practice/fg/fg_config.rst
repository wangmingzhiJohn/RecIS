fg配置
========

表示特征处理以及查表配置

标准常见使用
--------------------------------

String类型+Hash+查表
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

    {
        "features": [
            {
                "feature_name": "item_cate_id",
                "feature_type": "id_feature",
                "value_type": "String",
                "hash_bucket_size": 1000,
                "embedding_dimension": 32,
                "shared_name": "item_cate_id",
                "gen_key_type": "hash",
                "gen_val_type": "lookup"
            },
        ]
    }


Double类型+分桶+查表
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

    {
        "features": [
            {
                "feature_name": "uvsum",
                "feature_type": "raw_feature",
                "value_type": "Double",
                "embedding_dimension": 8,
                "shared_name": "uvsum",
                "boundaries": "0.0,2.0,3.0,5.0,6.0",
                "gen_key_type": "boundary",
                "gen_val_type": "lookup"
            }
        ]
    }

Double类型+直接使用
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

    {
        "features": [
            {
                "feature_name": "multimodal_correl_score",
                "feature_type": "raw_feature",
                "value_type": "Double",
                "value_dimension": 50,
                "gen_key_type": "idle",
                "gen_val_type": "idle"
            }
        ]
    }

String类型+MultiHash+查表（主搜RTP不支持）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

    {
        "features": [
            {
                "feature_name": "usersex_d_multihash",
                "feature_type": "id_feature",
                "value_type": "String",
                "hash_bucket_size": 10,
                "embedding_dimension": 8,
                "shared_name": "usersex_d",
                "compress_strategy": "yx:50,50,50,50:concat:4",
                "gen_key_type": "multihash",
                "gen_val_type": "multihash_lookup"
            }
        ]
    }

序列特征
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

    {
        "features": [
            {
                "feature_name": "cate_seq",
                "sequence_name": "cate_seq",
                "sequence_length": 50,
                "features": [
                    {
                        "feature_name": "cate_ids",
                        "feature_type": "id_feature",
                        "value_type": "String",
                        "hash_bucket_size": 1000,
                        "embedding_dimension": 32,
                        "shared_name": "cate_ids",
                        "gen_key_type": "hash",
                        "gen_val_type": "lookup"
                    },
                    {
                        "feature_name": "cate_rates",
                        "feature_type": "raw_feature",
                        "value_type": "Double",
                        "gen_key_type": "idle",
                        "gen_val_type": "idle"
                    }
                ]
            }
        ]
    }


特征来源于已有特征(特征clone)
--------------------------------

非序列特征
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

原始样本中只含有lp_time特征，新特征为lp_time_raw，并做Double类型+分桶+查表流程

.. code-block:: json

    {
        "features": [
            {
                "feature_name": "lp_time_raw",
                "from_feature": "lp_time",
                "feature_type": "raw_feature",
                "value_type": "Double",
                "embedding_dimension": 4,
                "shared_name": "longterm_time",
                "boundaries": "1.0,8564.0,16590.0",
                "gen_key_type": "boundary",
                "gen_val_type": "lookup"
            }
        ]
    }

序列特征
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. longpay_seq__context为前缀的序列特征，全部来源于longpay_seq为前缀的序列特征
2. 其中longpay_seq__context_lp_cnt特征，来源于longpay_seq_lp_cnt特征

.. code-block:: json

    {
        "features": [
            {
                "feature_name": "longpay_seq",
                "sequence_name": "longpay_seq__context",
                "sequence_length": 200,
                "features": [
                    {
                        "feature_name": "lp_cnt",
                        "from_feature": "longpay_seq_lp_cnt",
                        "feature_type": "raw_feature",
                        "value_type": "Double",
                        "embedding_dimension": 4,
                        "boundaries": "1.0,2.0,3.0,4.0",
                        "shared_name": "context_cnt",
                        "gen_key_type": "boundary",
                        "gen_val_type": "lookup"
                    }
                ]
            }
        ]
    }



配置完整字段解析
--------------------------------

非序列特征
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

    {
        "features": [
            {
                "feature_name": "特征名",
                "value_type": "特征值类型",
                "feature_type": "特征类型",
                "gen_key_type": "特征值处理类型",
                "gen_val_type": "特征值产出类型",
                "value_dimension": "特征值dim",
                "from_feature": "输入来源于io中的哪个特征",
                
                "hash_bucket_size": "哈希桶大小",
                "hash_type": "哈希方式",
                
                "compress_strategy": "multi哈希配置",
                
                "boundaries": "分桶值",

                "embedding_dimension": "查表dim",
                "shared_name": "表名",
                "combiner": "查表聚合方式",
                "trainable": "是否做训练梯度更新",
                "emb_device": "表所在的设备",
                "emb_type": "表类型",
                "admit_hook": "特征准入策略",
                "filter_hook": "特征过滤策略"
            },
        ]
    }

**通用配置**

- feature_name： 特征名称
- value_type： 特征值类型，可选 Integer | Double | String
- feature_type: 特征类型，可选 raw_feature | id_feature， raw_feature表示原始特征都已做过补齐操作，可以通过fixedlen_feature读取
- gen_key_type：特征值处理类型，可选 
    - idle: 不做处理，原值返回。（string类型特征的idle尚未支持）
    - boundary：分桶操作
    - hash：哈希处理
    - multihash: 先哈希再根据哈希值做multi哈希
    - mask：根据mask_value做mask（尚未支持）
- gen_val_type：特征输出方式，可选
    - idle：原值输出
    - lookup：查表后输出
    - multihash_lookup：用multihash结果查多个表，只有gen_key_type 为multihash时才被允许
- （可选）value_dimension：特征值维度，默认为1
- （可选）from_feature：处理的特征并不来源于io读取结果的feature_name，而是从其他特征拷贝出来

**哈希配置（gen_key_type为hash时需要）**

- hash_bucket_size：哈希操作分桶大小，0表示不做分桶
- （可选）hash_type：哈希方式，默认为farm。支持 farm ｜ murmur

**multi哈希配置（gen_key_type为multihash时需要）**

- compress_strategy：multihash的配置

.. code-block::

    配置形式为 
    ${prefix}:${bucket1},${bucket2},${bucket3},${bucket4}:${combiner}:${num}
    例子
    yx:50000,50000,50000,50000,concat,4

目前combiner仅支持concat，且数量一定是4

**分桶配置（gen_key_type为boundary时需要）**

- boundaries：分桶区间

**emb配置（gen_val_type为lookup时需要）**

- embedding_dimension：emb表的维度
- （可选）shared_name: emb表名，默认为feature_name
- （可选）combiner: emb聚合方式，默认为mean，可选 mean ｜ sum
- （可选）trainable: 是否做训练梯度更新，默认为true
- （可选）emb_type: 表类型，默认float32，默认根据fg统一配置处理，可选 float｜bf16 | fp16 | int8，默认float
- （可选）emb_device: 表所在的设备，默认根据fg统一配置处理，可选 cuda ｜ cpu，默认cuda
- （可选）admit_hook: 特征准入策略，默认无准入策略，示例配置 {"name": "ReadOnly"}
- （可选）filter_hook: 特征过滤策略，默认无过滤策略，示例配置 {"name": "GlobalStepFilter", "params": {"filter_step": 5000}}

序列特征
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

    {
        "features": [
            {
                "feature_name": "pv_item_seq",
                "sequence_length": 200,
                "sequence_name": "pv_item_seq",
                "features": [
                    "xxx": "xxx"
                ]
            },
        ]
    }

**序列配置**

- feature_name：序列前缀名
- sequence_name：序列特征处理后的前缀名，通常和feature_name一致
- sequence_length：序列最大长度
- features：序列下特征配置（注实际特征名为 序列前缀 + "_" + 序列下特征名），配置可参考上面非序列特征
