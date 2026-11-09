# 关于日志系统 (.clog)

`.clog` 日志不能直接用文本编辑器查看。先用 `pyclog` 命令行工具导出为 `.txt` 或 `.json` 文件，再打开查看。

模拟的日志由 `pyclog` 写入，采用二进制、分块压缩格式。好处是文件小，写入时对主循环的影响也小；代价是不能直接读，必须走导出这一步。

## 导出日志

```sh
python -m pyclog.cli --input <日志文件.clog> --output <输出文件名> --format <格式>
```

参数说明：

* `--input` / `-i`: 要读取的 `.clog` 文件路径。
* `--output` / `-o`: 导出文件的输出路径。
* `--format` / `-f`: 导出格式，`text`（默认）或 `json`。
* `--compress` / `-c`: 输出文件的压缩格式，`none`（默认）、`gzip` 或 `zstd`。

比如把 `simulation-2025-08-20_10-30-00.clog` 导出成纯文本：

```sh
python -m pyclog.cli --input simulation-2025-08-20_10-30-00.clog --output log_export.txt --format text
```

导出成 JSON：

```sh
python -m pyclog.cli -i simulation-2025-08-20_10-30-00.clog -o log_export.json -f json
```

纯文本适合直接看；JSON 保留了结构化字段，想自己写脚本筛事件时更方便。
