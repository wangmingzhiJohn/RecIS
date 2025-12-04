import json
from collections import OrderedDict

from recis.fg.utils import dict_lower_case


class MCParser:
    """Model Configuration parser for managing feature blocks and sequences.

    The MCParser class is responsible for parsing model configuration files
    and managing the organization of features into blocks. It handles both
    regular feature blocks and sequence blocks, providing utilities to check
    feature availability and manage feature groupings for model training.

    Key Features:
        - Parse JSON model configuration files
        - Manage feature blocks and sequence blocks
        - Filter features based on column usage requirements
        - Provide feature availability checking utilities
        - Support both file-based and dictionary-based configuration

    Attributes:
        uses_columns (list): List of column names to use. If None, uses all columns.
        mc_conf (dict): Parsed and formatted model configuration.
        seq_blocks (OrderedDict): Dictionary mapping sequence block names to feature names.
        blocks (OrderedDict): Dictionary of all usable feature names.
        fea_blocks (OrderedDict): Dictionary mapping block names to feature lists for concatenation.
    """

    def __init__(
        self, mc_config_path=None, mc_config=None, uses_columns=None, lower_case=False, with_seq_prefix=False
    ):
        """Initialize the MC Parser.

        Args:
            mc_config_path (str, optional): Path to the model configuration file.
                Either this or mc_config must be provided.
            mc_config (dict, optional): Model configuration dictionary.
                Either this or mc_config_path must be provided.
            uses_columns (list, optional): List of column names to use.
                If None, uses all columns from the configuration.
            lower_case (bool, optional): Whether to convert configuration keys
                to lowercase. Defaults to False.
            with_seq_prefix (bool, optional): Whether the feature name already has sequence block name as prefix.
                Defaults to False.

        Raises:
            AssertionError: If neither mc_config_path nor mc_config is provided.
        """
        self.uses_columns = uses_columns
        self.mc_conf = self._init_mc(mc_config_path, mc_config, lower_case)
        # sequence feature prefix name
        self.seq_blocks = OrderedDict()
        # usable feature name
        self.blocks = OrderedDict()
        # mc blocks for concat
        self.fea_blocks = OrderedDict()

        self.with_seq_prefix = with_seq_prefix

    def _init_mc(self, mc_config_path, mc_config, lower_case):
        """Initialize model configuration from file or dictionary.

        Args:
            mc_config_path (str): Path to the configuration file.
            mc_config (dict): Configuration dictionary.
            lower_case (bool): Whether to convert keys to lowercase.

        Returns:
            dict: Processed and formatted model configuration.

        Raises:
            AssertionError: If both mc_config_path and mc_config are None.
        """
        assert mc_config_path is not None or mc_config is not None, (
            "One of mc config file or mc config must be not none!"
        )
        if mc_config_path is not None:
            with open(mc_config_path) as f:
                mc_config = json.load(f)
        else:
            mc_config = mc_config
        mc_config = dict_lower_case(mc_config, lower_case)
        mc_config = self._format_mc(mc_config)
        return mc_config

    def _format_mc(self, mc_conf):
        """Format and filter model configuration based on column usage.

        This method processes the raw model configuration, filtering blocks
        based on the uses_columns setting and normalizing feature names.

        Args:
            mc_conf (dict): Raw model configuration dictionary.

        Returns:
            dict: Formatted model configuration with filtered blocks.
        """
        out_mc = {}
        for block_name, feas in mc_conf.items():
            if self.uses_columns is not None and block_name not in self.uses_columns:
                continue
            out_mc[block_name] = []
            for fea in feas:
                if isinstance(fea, dict):
                    fea = fea["column_name"]
                out_mc[block_name].append(fea)
        return out_mc

    @property
    def feature_blocks(self):
        """Get feature blocks dictionary.

        Returns:
            OrderedDict: Dictionary mapping block names to feature lists.
        """
        return self.fea_blocks

    @property
    def seq_block_names(self):
        """Get sequence block names.

        Returns:
            dict_keys: Keys of sequence blocks dictionary.
        """
        return self.seq_blocks.keys()

    def init_blocks(self, cand_seq_blocks):
        """Initialize feature blocks and sequence blocks.

        This method processes candidate sequence blocks and initializes
        the internal block structures for both sequence and regular features.

        Args:
            cand_seq_blocks (dict): Dictionary mapping candidate sequence block
                names to their corresponding feature names.
        """
        # get all sequence blocks
        for block_name, block_fea_name in cand_seq_blocks.items():
            if block_name in self.mc_conf:
                self.seq_blocks[block_name] = block_fea_name

        for block_name, features in self.mc_conf.items():
            # init feature blocks
            self.fea_blocks[block_name] = []
            prefix = ""
            if block_name in self.seq_blocks and not self.with_seq_prefix:
                prefix = block_name + "_"
            for fn in features:
                self.blocks[prefix + fn] = 1
                self.fea_blocks[block_name].append(prefix + fn)

    def has_fea(self, fea_name):
        """Check if a feature is available in the configuration.

        Args:
            fea_name (str): Name of the feature to check.

        Returns:
            bool: True if the feature is available, False otherwise.
        """
        return (fea_name in self.blocks) or (fea_name in self.seq_blocks)

    def has_seq_fea(self, seq_block, fea_name):
        """Check if a sequence feature is available in a sequence block.

        Args:
            seq_block (str): Name of the sequence block.
            fea_name (str): Name of the feature within the sequence block.

        Returns:
            bool: True if the sequence feature is available, False otherwise.
        """
        if seq_block not in self.seq_blocks:
            return False
        return (seq_block + "_" + fea_name) in self.blocks
