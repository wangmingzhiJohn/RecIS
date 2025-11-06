import json
import os

import torch

from recis.framework.filesystem import get_file_system
from recis.info import is_internal_enabled
from recis.nn.modules.hashtable import filter_out_sparse_param
from recis.serialize import Loader, Saver
from recis.utils.torch_fx_tool.export_torch_fx_tool import ExportTorchFxTool


if is_internal_enabled():
    from pangudfs_client.common.exception.exceptions import PanguException

    from recis.utils.mos import Mos
else:
    PanguException = None
    Mos = None


TMP_EXPORT_LOCAL_PATH = "./__tmp_export_path__/"


class Exporter:
    def __init__(
        self,
        model,
        sparse_model_name,
        dense_model_name,
        dataset,
        ckpt_dir,
        export_dir,
        dense_optimizer=None,
        export_folder_name="fx_user_model",
        export_model_name="user_model",
        export_outputs=None,
        fg=None,
        fg_conf_or_path=None,
        mc_conf_or_path=None,
        filter_sparse_opt=False,
    ):
        self.rank = int(os.environ.get("RANK", 0))
        self.shard_num = int(os.environ.get("WORLD_SIZE", 1))
        self.model = model
        self.sparse_model = model.get_submodule(sparse_model_name)
        self.sparse_model_name = sparse_model_name
        self.dense_model = model.get_submodule(dense_model_name)
        self.dense_model_name = dense_model_name
        self.dataset = dataset
        if ckpt_dir.startswith("model"):
            assert Mos is not None, "Cannot import mos, check interneal version."
            ckpt_dir = Mos(ckpt_dir, True).real_physical_path
        self.ckpt_dir = ckpt_dir
        if export_dir.startswith("model"):
            assert Mos is not None, "Cannot import mos, check interneal version."
            export_dir = Mos(export_dir, True).real_physical_path
        self.export_dir = export_dir

        self.dense_optimizer = dense_optimizer
        self.export_model_name = export_model_name
        self.export_outputs = export_outputs
        self.filter_sparse_opt = filter_sparse_opt

        assert fg is not None or fg_conf_or_path is not None, (
            "one of fg or fg_config must be not None"
        )
        assert fg is not None or mc_conf_or_path is not None, (
            "one of fg or mc_config must be not None"
        )
        if fg_conf_or_path is not None:
            if not isinstance(fg_conf_or_path, dict):
                with open(fg_conf_or_path) as f:
                    fg_conf_or_path = json.load(f)
            self.fg_conf = fg_conf_or_path
        else:
            self.fg_conf = fg.get_fg_conf()
        if mc_conf_or_path is not None:
            if not isinstance(mc_conf_or_path, dict):
                with open(mc_conf_or_path) as f:
                    mc_conf_or_path = json.load(f)
            self.mc_conf = mc_conf_or_path
        else:
            self.mc_conf = fg.get_mc_conf()

        self.fx_tool = ExportTorchFxTool(
            fx_folder=os.path.join(TMP_EXPORT_LOCAL_PATH, export_folder_name),
            model_name=self.export_model_name,
        )
        self.fx_tool.set_output_nodes_name(export_outputs)

    def export(self):
        self.prepare_model()
        self.export_sparse()
        self.export_dense()
        self.export_meta()

    def prepare_model(self):
        fs = get_file_system(self.export_dir)
        if not fs.exists(self.export_dir):
            try:
                fs.makedirs(self.export_dir + "/", exist_ok=True)
            except PanguException as e:
                if e.pangu_err_no == 7:
                    pass
        # load dense model
        pt_file = os.path.join(self.ckpt_dir, "model.pt")
        fs = get_file_system(pt_file)
        with fs.open(pt_file, "rb") as f:
            state_dict = torch.load(f=f)
        state_dict = {
            k.replace(f"{self.dense_model_name}.", ""): v for k, v in state_dict.items()
        }
        self.dense_model.load_state_dict(state_dict, strict=False)
        # maybe load sparse model
        if self.filter_sparse_opt:
            sparse_params = filter_out_sparse_param(self.sparse_model)
            loader = Loader(self.ckpt_dir, hashtables=sparse_params, tensors={})
            loader.load()

    def export_dense(self):
        # export dense to local tmp dir
        iterator = iter(self.dataset)
        stop_flag, data = next(iterator)
        dense_data = self.sparse_model(data)
        if self.dense_optimizer:
            self.dense_optimizer.zero_grad()
        self.fx_tool.export_fx_model(self.dense_model, dense_data[0], self.mc_conf)
        if self.rank == 0:
            # copy local tmp dir to dst dir
            fs = get_file_system(self.export_dir)
            fs.put(TMP_EXPORT_LOCAL_PATH, self.export_dir, recursive=True)

    def export_sparse(self):
        if not self.filter_sparse_opt:
            # get all sparse files (ends with `safetensors` or `json`)
            candidate_files = []
            fs = get_file_system(self.ckpt_dir)
            for file_name in fs.ls(self.ckpt_dir, detail=False):
                if file_name.endswith(("safetensors", "json")):
                    candidate_files.append(file_name)
            file_to_copy = []
            # partition files
            for i, file_name in enumerate(candidate_files):
                if i % self.shard_num == self.rank:
                    file_to_copy.append(file_name)
            # copy
            for file_name in file_to_copy:
                fs.copy(file_name, self.export_dir)
        else:
            # save sparse files
            sparse_params = filter_out_sparse_param(self.sparse_model)
            saver = Saver(
                shard_index=self.rank,
                shard_num=self.shard_num,
                parallel=1,
                hashtables=sparse_params,
                tensors={},
                path=self.export_dir,
            )
            saver.save()

    def export_meta(self):
        if self.rank == 0:
            fs = get_file_system(self.export_dir)
            # fg config
            with fs.open(os.path.join(self.export_dir, "fg.json"), "w") as out_f:
                json.dump(self.fg_conf, out_f)
