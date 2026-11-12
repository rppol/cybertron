import logging

import nncf
import openvino as ov
import torch
from openvino._offline_transformations import compress_model_transformation

log = logging.getLogger(__name__)


def convert(loaded, cfg) -> ov.Model:
    net = loaded.net
    example = loaded.tok(["an example input text", "a"], padding=True, return_tensors="pt")
    with torch.no_grad():
        model = ov.convert_model(net, example_input={n: example[n] for n in net.inputs})
    dims = ov.PartialShape([ov.Dimension(1, cfg.batch_size), ov.Dimension(1, loaded.max_len)])
    for port, name in zip(model.inputs, net.inputs):
        port.get_tensor().set_names({name})
    model.reshape({port: dims for port in model.inputs})
    if cfg.precision == "FP16":
        compress_model_transformation(model)
    elif cfg.precision == "INT8":
        model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
    return model


def compile(model, cfg):
    core = ov.Core()
    if cfg.cache_dir:
        core.set_property({"CACHE_DIR": str(cfg.cache_dir)})
    supported = {str(p) for p in core.get_property("CPU", "SUPPORTED_PROPERTIES")}
    compiled = core.compile_model(model, "CPU", properties(cfg, supported))
    info = {
        "precision": cfg.precision,
        "inference_precision": compiled.get_property("INFERENCE_PRECISION_HINT").to_string(),
        "hint": str(compiled.get_property("PERFORMANCE_HINT")),
        "streams": int(compiled.get_property("NUM_STREAMS")),
        "threads": int(compiled.get_property("INFERENCE_NUM_THREADS")),
        "optimal_requests": int(compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")),
        "inputs": [port.get_any_name() for port in compiled.inputs],
    }
    return compiled, info


def properties(cfg, supported):
    defaults = {"INFERENCE_PRECISION_HINT": "f32"} if cfg.precision == "FP32" else {}
    for key in defaults.keys() - supported:
        log.warning("CPU plugin does not support %s, leaving it unset", key)
    typed = {
        "PERFORMANCE_HINT": cfg.hint,
        "NUM_STREAMS": cfg.streams,
        "INFERENCE_NUM_THREADS": cfg.threads,
        "ENABLE_HYPER_THREADING": cfg.hyper_threading,
        "ENABLE_CPU_PINNING": cfg.cpu_pinning,
    }
    return (
        {k: v for k, v in defaults.items() if k in supported}
        | {k: v for k, v in typed.items() if v is not None}
        | cfg.ov_config
    )
