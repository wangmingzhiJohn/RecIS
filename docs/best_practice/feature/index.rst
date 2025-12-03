特征处理流程
=================

.. toctree::
   :maxdepth: 2
   
   feature
   engine

简单示例
----------

.. code-block:: python

    def get_feature_conf():
        feature_confs = []
        feature_confs.append(
            Feature("item_id").add_op(SelectField("item_id")).add_op(Hash("farm"))
        )
        feature_confs.append(
            Feature("user_id").add_op(SelectField("user_id")).add_op(Hash("farm"))
        )
        feature_confs.append(
            Feature("rate").add_op(SelectField("rate")).add_op(Bucketize([0, 0.1, 0.2]))
        )
        return feature_confs