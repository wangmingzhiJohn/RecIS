from dataclasses import dataclass
from typing import Optional

from recis.hooks.hook import Hook


@dataclass
class CheckpointSaveArguments:
    save_steps: Optional[int] = 100
    save_windows: Optional[int] = 1
    save_epochs: Optional[int] = None
    save_end: bool = True


@dataclass
class CheckpointLoadArguments:
    update_steps: Optional[int] = None
    update_windows: Optional[int] = 1
    update_epochs: Optional[int] = None


class CheckpointSaveHook(Hook):
    def __init__(
        self,
        saver,
        global_step,
        epoch,
        save_args: Optional[CheckpointSaveArguments] = None,
    ):
        super().__init__()
        self.saver = saver
        if save_args is None:
            save_args = CheckpointSaveArguments()
        self.args = save_args
        self.step = global_step
        self.window = 0
        self.epoch = epoch

    def window_mode(self, *args, **kwargs):
        self.args.save_steps = None
        self.args.save_end = False

    def after_step(
        self, label_key=None, label_value=None, is_train=True, *args, **kwargs
    ):
        if is_train:
            self.step += 1
            if (
                self.args.save_steps is not None
                and self.step % self.args.save_steps == 0
            ):
                ckpt_id = f"ckpt_{self.step}"
                self.saver.save(ckpt_id, label_key=label_key, label_value=label_value)

    def after_window(
        self, is_train=True, label_key=None, label_value=None, *args, **kwargs
    ):
        if is_train:
            self.window += 1
            if (
                self.args.save_windows is not None
                and self.window % self.args.save_windows == 0
            ):
                ckpt_id = f"ckpt_{self.step}"
                self.saver.save(ckpt_id, label_key=label_key, label_value=label_value)

    def after_epoch(
        self, is_train=True, label_key=None, label_value=None, *args, **kwargs
    ):
        if is_train:
            self.epoch += 1
            if (
                self.args.save_epochs is not None
                and self.epoch % self.args.save_epochs == 0
            ):
                ckpt_id = f"ckpt_{self.step}"
                self.saver.save(ckpt_id, label_key=label_key, label_value=label_value)

    def end(self, is_train=True, label_key=None, label_value=None, *args, **kwargs):
        if is_train and self.args.save_end:
            ckpt_id = f"ckpt_{self.step}"
            self.saver.save(ckpt_id, label_key=label_key, label_value=label_value)


class CheckpointLoadHook(Hook):
    def __init__(
        self,
        saver,
        global_step,
        epoch,
        load_args: Optional[CheckpointLoadArguments] = None,
    ):
        super().__init__()
        self.saver = saver
        if load_args is None:
            load_args = CheckpointLoadArguments()
        self.args = load_args
        self.gstep = global_step
        self.epoch = epoch
        self.step = 0
        self.window = 0

    def start(self, is_train=True, *args, **kwargs):
        self.saver.restore()

    def before_step(self, is_train=True, *args, **kwargs):
        if is_train:
            self.step += 1
            if (
                self.args.update_steps is not None
                and self.step % self.args.update_steps == 0
            ):
                self.saver.update_load()

    def before_window(self, is_train=True, *args, **kwargs):
        if is_train:
            self.window += 1
            if (
                self.args.update_windows is not None
                and self.window % self.args.update_windows == 0
            ):
                self.saver.update_load()
