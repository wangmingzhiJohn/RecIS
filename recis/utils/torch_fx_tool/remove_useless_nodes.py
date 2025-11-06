from typing import List, Set, Tuple

import torch

from recis.utils.logger import Logger


logger = Logger(__name__)


class RemoveUselessNodes:
    @staticmethod
    def _find_output_node(fx_graph_module):
        output_node = list(fx_graph_module.graph.nodes)[-1]
        if output_node.op != "output":
            for node in fx_graph_module.graph.nodes:
                if node.op == "output":
                    output_node = node
                    break
        if output_node.op != "output":
            raise ValueError("fx graph should have output node")
        return output_node

    @staticmethod
    def remove_useless_nodes(
        fx_graph_module, params_order, remove_useless_placeholders=False
    ) -> Tuple[torch.fx.GraphModule, List[str], List[str]]:
        """
        返回删除之后的graph
        重组之后的输入顺序
        无效的placeholder
        """
        invalid_placeholders = set()

        params_order_set = set(params_order)
        output_node = RemoveUselessNodes._find_output_node(fx_graph_module)
        valid_nodes: Set[torch.fx.Node] = set()
        stack = [output_node]

        while stack:
            current_node = stack.pop()
            if current_node not in valid_nodes:
                valid_nodes.add(current_node)
                stack.extend(current_node.all_input_nodes)

        impure_nodes = {
            node
            for node in fx_graph_module.graph.nodes
            if node.is_impure() and node not in valid_nodes and node.op != "placeholder"
        }

        if impure_nodes:
            valid_nodes.update(impure_nodes)
            for node in impure_nodes:
                logger.info(
                    f"Node '{node.name}' is impure and retained despite not being consumed."
                )
        invalid_nodes = [
            node for node in fx_graph_module.graph.nodes if node not in valid_nodes
        ]
        is_traveled = dict.fromkeys(invalid_nodes, False)

        def _delete_invalid_node(node):
            if node not in invalid_nodes or is_traveled[node]:
                return
            is_traveled[node] = True
            users_list = list(node.users.keys())
            for user in users_list:
                _delete_invalid_node(user)

            if node.op != "placeholder":
                try:
                    fx_graph_module.graph.erase_node(node)
                    logger.info(f"Erased node '{node.name}'.")
                except RuntimeError as e:
                    logger.warning(f"Failed to erase node '{node.name}': {e}")
            elif len(node.users) == 0:
                invalid_placeholders.add(node.name)
                params_order_set.remove(node.name)
                if remove_useless_placeholders:
                    fx_graph_module.graph.erase_node(node)
                    logger.warning(f"delete invalid placeholder: {node.name}")

        for node in invalid_nodes:
            if is_traveled[node]:
                continue
            _delete_invalid_node(node)

        fx_graph_module.recompile()

        params_order = [
            node_name for node_name in params_order if node_name in params_order_set
        ]

        return fx_graph_module, params_order, list(invalid_placeholders)
