


# 整合DSP与Elliposid中的过程记录
  ## G2O导致的segment fault
  很神奇！！ 修改g2o的编译类型为Debug。重新编译后就能运行了。
  ## 

# G2O的编译指令
cmake -DEigen3_DIR="/home/robotlab/thirdparty/for_dspslam/eigen/install/share/eigen3/cmake" ..