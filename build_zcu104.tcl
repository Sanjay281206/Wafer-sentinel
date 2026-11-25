# WaferSentinel accelerator on the ZCU104 (XCZU7EV-2FFVC1156): builds the block design, bitstream and XSA.
#
#   python scripts/prepare_build.py --export fpga_export --out build
#   vivado -mode batch -source vivado/build_zcu104.tcl -tclargs <abs path of build/> [pl_clk_mhz=100] [jobs=8]
#
# Block design:  Zynq UltraScale+ PS (ZCU104 board preset) --M_AXI_HPM0_FPD--> SmartConnect --> wafer_cnn_axil (RTL module)
#                pl_clk0 (pl_clk_mhz) clocks everything, proc_sys_reset gives the active-low reset.
# Needs the ZCU104 board files (shipped with Vivado: Tools > Vivado Store > Boards, or check  get_board_parts *zcu104* ).
# NOT tested by me in Vivado (no Xilinx tools where this package was written): every step prints what it did; if a line
# fails, the guide's troubleshooting section lists the usual causes, and the GUI steps in the guide do the same thing.

set build    [file normalize [lindex $argv 0]]
set clk_mhz  [expr {[llength $argv] > 1 ? [lindex $argv 1] : 100}]
set jobs     [expr {[llength $argv] > 2 ? [lindex $argv 2] : 8}]
set proj_dir "$build/vivado_proj"
set out_dir  "$build/output"
foreach f {wafer_cnn_accel.v wafer_cnn_axil.v} {
    if {![file exists "$build/rtl/$f"]} { error "$build/rtl/$f not found: run scripts/prepare_build.py first" }
}
file mkdir $out_dir

proc latest_ip {name} {
    set d [lsort -dictionary [get_ipdefs -filter "NAME == $name"]]
    if {[llength $d] == 0} { error "IP $name not found in this Vivado installation" }
    return [lindex $d end]
}

# ---------------------------------------------------------------- project
create_project -force wafer_zcu104 $proj_dir -part xczu7ev-ffvc1156-2-e
set bp [get_board_parts -quiet *zcu104*]
if {[llength $bp] == 0} { error "No ZCU104 board files found. Install them (Vivado Store > Boards) and re-run." }
set_property board_part [lindex [lsort -dictionary $bp] end] [current_project]
puts "INFO: board part [get_property board_part [current_project]]"
add_files -norecurse [list "$build/rtl/wafer_cnn_accel.v" "$build/rtl/wafer_cnn_axil.v"]
update_compile_order -fileset sources_1

# ---------------------------------------------------------------- block design
create_bd_design design_1
create_bd_cell -type ip -vlnv [latest_ip zynq_ultra_ps_e] zynq_ultra_ps_e_0
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} [get_bd_cells zynq_ultra_ps_e_0]
# one AXI master (HPM0_FPD) for the accelerator registers, PL clock 0 at the requested frequency
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $clk_mhz] [get_bd_cells zynq_ultra_ps_e_0]

create_bd_cell -type module -reference wafer_cnn_axil wafer_cnn_axil_0
create_bd_cell -type ip -vlnv [latest_ip proc_sys_reset] proc_sys_reset_0
create_bd_cell -type ip -vlnv [latest_ip smartconnect] axi_smc
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells axi_smc]

connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] [get_bd_intf_pins axi_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M00_AXI] [get_bd_intf_pins wafer_cnn_axil_0/s_axi]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] \
    [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk] [get_bd_pins axi_smc/aclk] \
    [get_bd_pins wafer_cnn_axil_0/s_axi_aclk] [get_bd_pins proc_sys_reset_0/slowest_sync_clk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins proc_sys_reset_0/ext_reset_in]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
    [get_bd_pins axi_smc/aresetn] [get_bd_pins wafer_cnn_axil_0/s_axi_aresetn]

assign_bd_address
set accel_base "unknown"
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces zynq_ultra_ps_e_0/Data]] {
    if {[string match *wafer_cnn_axil_0* $seg]} {
        set accel_base [get_property OFFSET $seg]
        puts "INFO: accelerator [get_property NAME $seg]  offset $accel_base  range [get_property RANGE $seg]"
    }
}
if {$accel_base ne "0x00A0000000" && $accel_base ne "0xA0000000"} {
    puts "WARNING: accelerator base is $accel_base, not 0xA0000000: pass it to zcu_server.py --base and ACCEL_BASE in main.c"
}
catch {puts "INFO: PL clock 0 actual frequency: [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ [get_bd_cells zynq_ultra_ps_e_0]] MHz"}

validate_bd_design
save_bd_design
generate_target all [get_files design_1.bd]

# ---------------------------------------------------------------- wrapper, implementation, bitstream
set wrapper [make_wrapper -files [get_files design_1.bd] -top]
if {$wrapper eq ""} { set wrapper [lindex [glob -nocomplain $proj_dir/*/sources_1/bd/design_1/hdl/design_1_wrapper.v] 0] }
add_files -norecurse $wrapper
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "implementation did not finish: see $proj_dir/wafer_zcu104.runs/impl_1/runme.log" }

open_run impl_1
report_utilization    -file "$out_dir/utilization.txt"
report_timing_summary -file "$out_dir/timing_summary.txt"
report_power          -file "$out_dir/power_estimate.txt"
set wns [get_property STATS.WNS [get_runs impl_1]]
set whs [get_property STATS.WHS [get_runs impl_1]]
puts "INFO: worst negative slack $wns ns, worst hold slack $whs ns  ->  [expr {($wns >= 0 && $whs >= 0) ? {TIMING MET} : {TIMING NOT MET: lower pl_clk_mhz}}]"

write_hw_platform -fixed -include_bit -force -file "$out_dir/wafer_zcu104.xsa"
file copy -force [get_property DIRECTORY [get_runs impl_1]]/design_1_wrapper.bit "$out_dir/wafer_zcu104.bit"
set hwh [lindex [glob -nocomplain $proj_dir/*/sources_1/bd/design_1/hw_handoff/design_1.hwh] 0]
if {$hwh ne ""} { file copy -force $hwh "$out_dir/wafer_zcu104.hwh" } else { puts "WARNING: design_1.hwh not found: search $proj_dir for it and copy it as wafer_zcu104.hwh (PYNQ needs it)" }
puts "DONE. Files for the board are in $out_dir : wafer_zcu104.bit, wafer_zcu104.hwh, wafer_zcu104.xsa, reports."
