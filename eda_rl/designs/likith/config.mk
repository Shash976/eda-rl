export DESIGN_NAME = id
export DESIGN_NICKNAME = id
export PLATFORM = asap7


##################################################
# RTL
##################################################

export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/id.v

##################################################
# Constraints
##################################################

export SDC_FILE = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/constraint.sdc

##################################################
# FastRoute
##################################################

export FASTROUTE_TCL = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/fastroute.tcl

##################################################
# Floorplan
##################################################

export DIE_AREA  = 0 0 15 15
export CORE_AREA = 1 1 14 14

##################################################
# Placement
##################################################

export PLACE_DENSITY = 0.65

##################################################
# Synthesis
##################################################

export SYNTH_STRATEGY  = DELAY 2
export SYNTH_BUFFERING = 1
export SYNTH_SIZING    = 1

##################################################
# Global Placement
##################################################

export GPL_TIMING_DRIVEN      = 1
export GPL_ROUTABILITY_DRIVEN = 0

##################################################
# CTS
##################################################

export CTS_CLUSTER_SIZE     = 20
export CTS_CLUSTER_DIAMETER = 100

##################################################
# Resizer
##################################################

export REPAIR_ANTENNAS = 1
export RECOVER_POWER   = 0


export SKIP_CTS_REPAIR_TIMING = 1
