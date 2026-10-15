"""对外可展示的启动问题；详细异常保留在服务端日志。"""


class StartupIssue(RuntimeError):
    def __init__(self, code, component, message, hint):
        super().__init__(message)
        self.code, self.component = code, component
        self.message, self.hint = message, hint

    def as_dict(self):
        return {"code": self.code, "component": self.component,
                "message": self.message, "hint": self.hint}


def check_model_files(path, component):
    from pathlib import Path
    directory = Path(path)
    weights = any(directory.glob("*.safetensors")) or any(directory.glob("pytorch_model*.bin"))
    if not directory.is_dir() or not (directory / "config.json").is_file() or not weights:
        raise StartupIssue("model_files_missing", component,
                           f"{component} 模型文件缺失或不完整",
                           f"检查 model/ 挂载，并上传完整的 {directory.name} 模型目录。")
