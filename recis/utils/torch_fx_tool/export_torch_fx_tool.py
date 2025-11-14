import copy
import json
import os
import pickle
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


try:
    from torchrec.distributed.model_parallel import (
        DistributedDataParallel,
        DistributedModelParallel,
        FullyShardedDataParallel,
    )

    TORCHREC_INSTALLED = True
except ImportError:
    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.nn.parallel import DistributedDataParallel

    TORCHREC_INSTALLED = False

from .remove_useless_nodes import RemoveUselessNodes


try:
    parts = torch.__version__.split(".")
    torch_major_version = int(parts[0])
    torch_minor_version = int(parts[1])
except (ValueError, IndexError):
    torch_major_version = None
    torch_minor_version = None


if torch_minor_version is not None and (torch_major_version, torch_minor_version) > (
    2,
    7,
):
    # Utilize torch.export.save implementation in PyTorch 2.7 for versions higher than 2.7.
    # Remove this once RTP adds support for torch.export.load in PyTorch versions above 2.7.

    import zipfile

    from torch._export.serde.schema import SCHEMA_VERSION
    from torch._export.serde.serialize import SerializedArtifact, serialize
    from torch.export import DEFAULT_PICKLE_PROTOCOL, ExportedProgram
    from torch.types import FileLike

    def torch_export_save(
        ep: ExportedProgram,
        f: FileLike,
        *,
        extra_files: Optional[dict[str, Any]] = None,
        opset_version: Optional[dict[str, int]] = None,
        pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
    ) -> None:
        if not isinstance(ep, ExportedProgram):
            raise TypeError(
                f"The 'ep' parameter must be an instance of 'ExportedProgram', got '{type(ep).__name__}' instead."
            )

        artifact: SerializedArtifact = serialize(ep, opset_version, pickle_protocol)

        if isinstance(f, (str, os.PathLike)):
            f = os.fspath(f)

        with zipfile.ZipFile(f, "w") as zipf:
            # Save every field in the SerializedArtifact to a file.
            assert isinstance(artifact.exported_program, bytes)
            zipf.writestr("serialized_exported_program.json", artifact.exported_program)
            zipf.writestr("serialized_state_dict.pt", artifact.state_dict)
            zipf.writestr("serialized_constants.pt", artifact.constants)
            zipf.writestr("serialized_example_inputs.pt", artifact.example_inputs)

            zipf.writestr("version", ".".join(map(str, SCHEMA_VERSION)))

            # Add extra files if provided
            if extra_files:
                for extra_file_name, content in extra_files.items():
                    encoded_content = content.encode("utf-8")
                    zipf.writestr(f"extra_files/{extra_file_name}", encoded_content)
else:
    torch_export_save = torch.export.save


class ExportTorchFxTool:
    def __init__(
        self,
        fx_folder: str = "./fx_user_model",
        model_name: str = "user_model",
        dynamic: bool = True,
        device="cuda",
        max_batch_size=204800,
    ):
        self.fx_folder = fx_folder
        self.model_name = model_name
        self.device = torch.device(device)
        self.graph_module = None
        self.inputs_dict = None
        self.params_order = None
        self.output_nodes_name = None
        self.dynamic = dynamic
        self.max_batch_size = max_batch_size
        # 保存的batch size的数据过大 会造成离线表和biz构建过程很慢 甚至有显存不够的风险
        self.cut_off_batch_size = 128
        self.EXPORTED_MODEL_NAME = "exported_model.pt"
        os.makedirs(self.fx_folder, exist_ok=True)

    @staticmethod
    def set_exported_mode(model: nn.Module):
        model.eval()
        for module in model.modules():
            # 如果该模块有'exported'属性，则设为True
            if hasattr(module, "exported"):
                module.exported = True

    @staticmethod
    def set_train_mode(model: nn.Module):
        model.train()
        for module in model.modules():
            # 如果该模块有'exported'属性，则设为False
            if hasattr(module, "exported"):
                module.exported = False

    def _get_unwrapped_module(self, module: nn.Module) -> nn.Module:
        """
        Unwraps module wrapped by DMP, DDP, or FSDP.
        """
        if not TORCHREC_INSTALLED:
            wrappers = (DistributedDataParallel, FullyShardedDataParallel)
        else:
            wrappers = (
                DistributedDataParallel,
                FullyShardedDataParallel,
                DistributedModelParallel,
            )

        while isinstance(module, wrappers):
            if TORCHREC_INSTALLED and isinstance(module, DistributedModelParallel):
                module = module._dmp_wrapped_module
            elif isinstance(module, FullyShardedDataParallel):
                module = module._fsdp_wrapped_module
            else:
                module = module.module
        return module

    def _validate(self):
        # start_time = time.perf_counter()
        assert self.graph_module is not None, "please export the fx model first!"
        with torch.no_grad():
            self.user_model.eval()
            output_base = self.user_model(copy.deepcopy(self.original_inputs_dict))

        if self.output_nodes_name is not None:
            if (
                isinstance(output_base, torch.Tensor)
                and len(self.output_nodes_name) != 1
            ):
                raise ValueError(
                    "model only have one output, however you set multi output_nodes_name"
                )
            if isinstance(output_base, tuple) and len(self.output_nodes_name) != len(
                output_base
            ):
                raise ValueError(
                    "output_nodes_name must be the same length as real user_model_output"
                )

        # 保存输出节点的信息
        output_node_info = dict()
        if isinstance(output_base, torch.Tensor):
            output_node_info["OutputNodes"] = []
            output_node_info["OutputNodes"].append(
                {
                    "OutputNodeName": "torch_output_" + str(0)
                    if self.output_nodes_name is None
                    else self.output_nodes_name[0],
                    "dtype": str(output_base.dtype),
                    "shape": list(output_base.shape),
                }
            )
        elif isinstance(output_base, tuple):
            output_node_info["OutputNodes"] = []
            for i in range(len(output_base)):
                output_node_info["OutputNodes"].append(
                    {
                        "OutputNodeName": "torch_output_" + str(i)
                        if self.output_nodes_name is None
                        else self.output_nodes_name[i],
                        "dtype": str(output_base[i].dtype),
                        "shape": list(output_base[i].shape),
                    }
                )
        else:
            raise ValueError(
                "user_model_output must be torch.Tensor or tuple of torch.Tensor"
            )
        json.dump(
            output_node_info,
            open(os.path.join(self.fx_folder, "output_info.json"), "w"),
            indent=4,
        )
        with torch.inference_mode():
            output = self.graph_module(**self.inputs_dict)
        if isinstance(output, torch.Tensor):
            if torch.allclose(output_base, output, atol=1e-6):
                return True
            else:
                return False
        elif isinstance(output, tuple):
            all_close_flag = True
            for i, j in zip(output, output_base):
                all_close_flag &= torch.allclose(i, j, atol=1e-6)
            return all_close_flag
        else:
            raise ValueError(
                "dynamicLib_output must be torch.Tensor or tuple of torch.Tensor"
            )

    def set_output_nodes_name(self, output_nodes_name: List[str]):
        """
        指定输出节点的名称
            例如 ["ctr", "cvr"] 且必须与网络真实的输出节点数量一致
            未指定 默认是 ["torch_output_0", "torch_output_1"]
        """
        self.output_nodes_name = output_nodes_name

    def _dump(self, mc_config):
        if not os.path.exists(os.path.join(self.fx_folder, "inputs_dict_lite.json")):
            with open(
                os.path.join(self.fx_folder, "inputs_dict_lite.json"),
                "w",
                encoding="utf-8",
            ) as json_file:
                mark_dict = {
                    name: {
                        "idx": idx,
                        "shape": list(self.inputs_dict[name].shape),
                        "dtype": str(self.inputs_dict[name].dtype),
                    }
                    for idx, name in enumerate(sorted(self.inputs_dict.keys()))
                }
                json.dump(mark_dict, json_file, indent=4)

        if not os.path.exists(os.path.join(self.fx_folder, "inputs_dict.json")):
            with open(
                os.path.join(self.fx_folder, "inputs_dict.json"), "w", encoding="utf-8"
            ) as json_file:
                mark_dict = {
                    name: {
                        "idx": idx,
                        "shape": list(self.inputs_dict[name].shape),
                        "dtype": str(self.inputs_dict[name].dtype),
                        "values": self.inputs_dict[name].tolist(),
                    }
                    for idx, name in enumerate(sorted(self.inputs_dict.keys()))
                }
                json.dump(mark_dict, json_file, indent=4)

        if not os.path.exists(os.path.join(self.fx_folder, "inputs_dict.pkl")):
            pickle.dump(
                self.inputs_dict,
                open(os.path.join(self.fx_folder, "inputs_dict.pkl"), "wb"),
            )

        if not os.path.exists(os.path.join(self.fx_folder, "input_columns.json")):
            with open(
                os.path.join(self.fx_folder, "input_columns.json"),
                "w",
                encoding="utf-8",
            ) as json_file:
                if mc_config is None:
                    mc_config = self.user_model.mc_config
                json.dump(mc_config, json_file, indent=4)

        if not os.path.exists(os.path.join(self.fx_folder, "params_order.json")):
            with open(
                os.path.join(self.fx_folder, "params_order.json"), "w", encoding="utf-8"
            ) as json_file:
                json.dump(self.params_order, json_file, indent=4)

    def export_fx_model(
        self,
        user_model: nn.Module,
        input: Dict[str, torch.Tensor],
        mc_config: Optional[Dict[str, str]] = None,
    ):
        """export model to a FX Graph
        args:
          input: input of the user_model
        """

        # _transformer_encoder_layer_fwd算子不能被dynamo支持 禁用fastpath进行关闭
        fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)

        def _checkInputCls(tensor_dict):
            # 输入input必须严格是torch.Tensor类型
            # 并且第一维需要一致
            batch_list = list()
            for key in tensor_dict.keys():
                assert isinstance(tensor_dict[key], torch.Tensor), (
                    "input must be torch.Tensor"
                )
                batch_list.append(tensor_dict[key].shape[0])
            if self.dynamic:
                # 所有tensor的第一维需要一致
                assert len(set(batch_list)) == 1, "input must be the same batch_size"

        def _checkZeroGrad(model):
            # 导出graph前需要清空梯度
            for param in model.parameters():
                if param.grad is not None and param.grad.abs().sum() != 0:
                    raise ValueError("model need zero grad before export fx model")

        def _modifyBatchSize(tensor_dict):
            # 输入input的batch_size必须小于等于self.max_batch_size
            # 如果不等 则进行切片
            for key in tensor_dict.keys():
                if tensor_dict[key].shape[0] > min(
                    self.max_batch_size, self.cut_off_batch_size
                ):
                    tensor_dict[key] = tensor_dict[key][
                        : min(self.max_batch_size, self.cut_off_batch_size)
                    ]

        _checkInputCls(input)
        if self.dynamic:
            _modifyBatchSize(input)

        self.user_model = user_model
        wrappers = (DistributedDataParallel, FullyShardedDataParallel)
        if TORCHREC_INSTALLED:
            wrappers = (
                DistributedDataParallel,
                FullyShardedDataParallel,
                DistributedModelParallel,
            )
        if isinstance(user_model, wrappers):
            self.user_model = self._get_unwrapped_module(user_model)

        _checkZeroGrad(user_model)

        self.user_model = self.user_model.to(self.device)
        self.original_inputs_dict = {
            tensorName: copy.deepcopy(input[tensorName].detach()).to(self.device)
            for tensorName in sorted(input.keys())
        }
        self.inputs_dict = {
            tensorName: copy.deepcopy(input[tensorName].detach()).to(self.device)
            for tensorName in sorted(input.keys())
        }
        self.inputs_list = [
            self.inputs_dict[key] for key in sorted(self.inputs_dict.keys())
        ]
        self.params_order = list(self.inputs_dict.keys())

        def add_dynamic_forward(cls):
            """
            动态绑定forward函数的装饰器
            """
            args = sorted(self.inputs_dict.keys())
            func_signature = ", ".join(args)
            dict_creation_lines = [f"        '{arg}': {arg}," for arg in args]
            func_body = (
                "    inputs = {{\n" + "\n".join(dict_creation_lines) + "\n    }\n"
            )
            func_body += "    return self.user_model(inputs)\n"
            func_code = f"def forward(self, {func_signature}):\n{func_body}"
            # print("Generated forward method code:\n", func_code)

            # 执行代码字符串在当前作用域中定义forward方法
            local_vars = {}
            exec(func_code, globals(), local_vars)

            # 绑定生成的 forward 方法到类
            cls.forward = local_vars["forward"]
            return cls

        @add_dynamic_forward
        class FxWarpModel(nn.Module):
            """
            原始的user_model的输入是一个字典Dict[str, Tensor]
            aot出来的so的签名必须是Tensor的形式
            为了避免修改用户代码 对用户的model进行warp
            FxWarpModel动态生成forward函数
            """

            def __init__(self, user_model: nn.Module) -> None:
                super().__init__()
                self.user_model = user_model

        def create_remove_placeholder_warp_model(
            graph_model, invalid_placeholder, inputs_dict
        ):
            """
            工厂函数 创建带有动态forward方法的RemovePlaceholderWarpModel
            """
            # 获取有效的输入参数
            valid_args = [
                arg
                for arg in sorted(inputs_dict.keys())
                if arg not in invalid_placeholder
            ]

            def add_dynamic_forward(cls):
                """
                动态绑定forward函数的装饰器
                """
                # 构建函数签名
                func_signature = ", ".join(valid_args)

                # 构建函数体
                dict_creation_lines = [f"        '{arg}': {arg}," for arg in valid_args]
                func_body = (
                    "    inputs = {{\n" + "\n".join(dict_creation_lines) + "\n    }\n"
                )

                # 为invalid_placeholder添加空tensor
                for ph in invalid_placeholder:
                    func_body += f"    inputs['{ph}'] = torch.empty_like(self.inputs_dict['{ph}'])\n"

                func_body += "    return self.graph_model(**inputs)\n"

                # 完整的函数代码
                func_code = f"def forward(self, {func_signature}):\n{func_body}"

                # 执行代码字符串
                local_vars = {}
                exec(func_code, globals(), local_vars)

                # 绑定生成的forward方法到类
                cls.forward = local_vars["forward"]
                return cls

            @add_dynamic_forward
            class RemovePlaceholderWarpModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.graph_model = graph_model
                    self.invalid_placeholder = invalid_placeholder
                    self.inputs_dict = copy.deepcopy(inputs_dict)

            return RemovePlaceholderWarpModel()

        warp_model = FxWarpModel(self.user_model)
        ExportTorchFxTool.set_exported_mode(warp_model)

        # 第一维是batch维度默认可变
        if self.dynamic:
            dynamic_shapes = {
                input_name: {
                    0: torch.export.Dim("batch", min=1, max=self.max_batch_size)
                }
                for input_name in self.inputs_dict.keys()
            }
            with torch.no_grad():
                exported_program_model: torch.export.ExportedProgram = (
                    torch.export.export(
                        warp_model,
                        args=(),
                        kwargs=self.inputs_dict,
                        dynamic_shapes=dynamic_shapes,
                    )
                )
        else:
            with torch.no_grad():
                exported_program_model: torch.export.ExportedProgram = (
                    torch.export.export(warp_model, args=(), kwargs=self.inputs_dict)
                )

        graph_module = exported_program_model.module()
        # 由于PyTorch fx.GraphModule自身in_spec的约束
        # 删除placeholder没有直接的方法 需要使用warpModel进行取巧的删除
        self.graph_module, self.params_order, invalid_placeholders = (
            RemoveUselessNodes.remove_useless_nodes(
                graph_module, self.params_order, remove_useless_placeholders=False
            )
        )
        remove_ph_warp_model = create_remove_placeholder_warp_model(
            self.graph_module, invalid_placeholders, self.inputs_dict
        )

        # refresh inputs_dict
        self.inputs_dict = {p: self.inputs_dict[p] for p in self.params_order}
        self.inputs_list = [self.inputs_dict[p] for p in self.params_order]

        if self.dynamic:
            dynamic_shapes = {
                input_name: {
                    0: torch.export.Dim("batch", min=1, max=self.max_batch_size)
                }
                for input_name in self.inputs_dict.keys()
            }
            with torch.no_grad():
                exported_program_model: torch.export.ExportedProgram = (
                    torch.export.export(
                        remove_ph_warp_model,
                        args=(),
                        kwargs=self.inputs_dict,
                        dynamic_shapes=dynamic_shapes,
                    )
                )
        else:
            with torch.no_grad():
                exported_program_model: torch.export.ExportedProgram = (
                    torch.export.export(
                        remove_ph_warp_model, args=(), kwargs=self.inputs_dict
                    )
                )

        # exported_program_model中已经不包含不需要的placeholder 转换为了内部的get_attr节点 可以直接进行删除
        graph_module = exported_program_model.module()
        self.graph_module, self.params_order, invalid_placeholders = (
            RemoveUselessNodes.remove_useless_nodes(
                graph_module, self.params_order, remove_useless_placeholders=True
            )
        )
        # refresh inputs_dict
        self.inputs_dict = {p: self.inputs_dict[p] for p in self.params_order}
        self.inputs_list = [self.inputs_dict[p] for p in self.params_order]
        # 再度转换为ExportedModule
        if self.dynamic:
            dynamic_shapes = {
                input_name: {
                    0: torch.export.Dim("batch", min=1, max=self.max_batch_size)
                }
                for input_name in self.inputs_dict.keys()
            }
            with torch.no_grad():
                exported_program_model: torch.export.ExportedProgram = (
                    torch.export.export(
                        self.graph_module,
                        args=(),
                        kwargs=self.inputs_dict,
                        dynamic_shapes=dynamic_shapes,
                    )
                )
        else:
            with torch.no_grad():
                exported_program_model: torch.export.ExportedProgram = (
                    torch.export.export(
                        self.graph_module, args=(), kwargs=self.inputs_dict
                    )
                )

        torch_export_save(
            exported_program_model,
            os.path.join(self.fx_folder, self.EXPORTED_MODEL_NAME),
        )
        self.graph_module = exported_program_model.module()

        # torch.fx.symbolic_trace api acquire fx-level graph
        fx_graph_module: torch.fx.GraphModule = torch.fx.symbolic_trace(
            self.graph_module
        )
        fx_graph_module.to_folder(
            os.path.join(self.fx_folder, self.model_name), self.model_name
        )

        graph_file_path = os.path.join(self.fx_folder, "graph.txt")
        with open(graph_file_path, "w", encoding="utf-8") as file:
            print(fx_graph_module.graph, file=file)
            print(fx_graph_module.code, file=file)

        self._dump(mc_config)

        if not self._validate():
            raise ValueError("export_fx_output is not equal to user_model_output!")

        output_base = self.user_model(copy.deepcopy(self.original_inputs_dict))
        pickle.dump(
            output_base, open(os.path.join(self.fx_folder, "output_base.pkl"), "wb")
        )

        # 恢复fastpath
        torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
        ExportTorchFxTool.set_train_mode(warp_model)
