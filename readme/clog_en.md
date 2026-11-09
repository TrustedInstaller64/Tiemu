# About the Logging System (.clog)

You can't read a `.clog` file in a text editor. Export it first with the `pyclog` command-line tool to `.txt` or `.json`, then open the result.

The simulation writes its log through `pyclog`, which uses a binary, chunked and compressed format. Files stay small and the write path stays cheap; the trade-off is that you always need the export step.

## Exporting a log

```sh
python -m pyclog.cli --input <your_log_file.clog> --output <output_filename> --format <format>
```

Arguments:

* `--input` / `-i`: path to the `.clog` file.
* `--output` / `-o`: path for the exported file.
* `--format` / `-f`: `text` (default) or `json`.
* `--compress` / `-c`: compression for the output file: `none` (default), `gzip` or `zstd`.

For example, to turn `simulation-2025-08-20_10-30-00.clog` into plain text:

```sh
python -m pyclog.cli --input simulation-2025-08-20_10-30-00.clog --output log_export.txt --format text
```

Or into JSON:

```sh
python -m pyclog.cli -i simulation-2025-08-20_10-30-00.clog -o log_export.json -f json
```

Plain text is easiest to read; JSON keeps the structured fields, which is handy if you want to filter events with a script.
