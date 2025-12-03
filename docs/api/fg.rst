Feature Generator
===========================

RecIS's Feature Generator module provides configurable feature and embedding settings.


FG
---------------

.. currentmodule:: recis.fg.feature_generator

.. autoclass:: FG
   :members: __init__, feature_blocks, seq_block_names, sample_ids, labels, feature_shapes, block_shapes, is_seq_block, get_block_seq_len, add_label, add_id, get_shape, add_io_features, get_emb_confs, get_feature_confs

.. autofunction:: build_fg

FGParser
---------------

.. currentmodule:: recis.fg.fg_parser

.. autoclass:: FGParser
   :members: __init__, feature_blocks, io_configs, emb_configs, seq_block_names, get_seq_len

MCParser
---------------

.. currentmodule:: recis.fg.mc_parser

.. autoclass:: MCParser
   :members: __init__, feature_blocks, seq_block_names, has_fea, has_seq_fea