export DESIGN_NAME = alu_4bit
export DESIGN_NICKNAME = alu
export PLATFORM = sky130hd

export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/alu_4bit.v
export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/constraint.sdc

export DIE_AREA = 0 0 60 60
export CORE_AREA = 5 5 55 55

export PLACE_DENSITY_LB_ADDON = 0.30

export TNS_END_PERCENT = 100

export REMOVE_ABC_BUFFERS = 1

export SKIP_CTS_REPAIR_TIMING = 1
