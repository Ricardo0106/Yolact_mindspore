# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
source /root/miniconda3/bin/activate ci3.7
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=/usr/local/lib64/:${LOCAL_ASCEND}/add-ons/:${LOCAL_ASCEND}/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/fwkacllib/ccec_compiler/bin/:${PATH}                 # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                 # Python library that TBE implementation depess

source /root/miniconda3/bin/activate dsj2
