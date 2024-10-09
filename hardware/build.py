import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from custom_steps import *

build_dataflow_steps = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    custom_step_streamline,
    custom_step_convert_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_minimize_bit_width",
    "step_apply_folding_config",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "step_measure_rtlsim_performance",
    "step_out_of_context_synthesis",
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package",
]

def main(model, output, folding_config_file):
    cfg = build.DataflowBuildConfig(
    output_dir          = output,
    synth_clk_period_ns = 3.3,
    target_fps          = 30,
    board               = "TySOM-3A-ZU19EG",
    shell_flow_type     = build_cfg.ShellFlowType.VIVADO_ZYNQ,
    steps               = build_dataflow_steps,
    folding_config_file = folding_config_file,
    split_large_fifos = True,
    auto_fifo_depths = False,
    large_fifo_mem_style = build_cfg.LargeFIFOMemStyle.URAM,
    verbose = False,
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ]
    )
    build.build_dataflow_cfg(model, cfg)
    return

if __name__ == '__main__':
    model="models/p-enet.onnx"
    output="output"
    folding="configs/p-enet_m_config.json"
    main(model, output, folding)
