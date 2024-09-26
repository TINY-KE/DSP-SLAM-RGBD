[DSP与Elliposid整合]


# 整合DSP与Elliposid中的过程记录
  ## G2O导致的segment fault
  很神奇！！ 修改g2o的编译类型为Debug。重新编译后就能运行了。
  ## 修改python文件，导出bbox
    + 获取bbox的位置：
      masks_2d = det_2d["pred_masks"]
      bboxes_2d = det_2d["pred_boxes"]
    + 运行get_detections(self)之后， Frame类的self.instances包含了最显著的一个 物体检测结果
    + Frame的instances 又转存到了 MonoSequence的detections_in_current_frame
       self.detections_in_currentdetections_in_current_frame_frame = self.current_frame.instances
    
  ## python中物体检测对检测框的筛选
    + 不能太靠近边缘
       Remove those on the margin
        cond1 = (boxes[:, 0] >= self.mEdge) \
                & (boxes[:, 1] > self.mEdge) \
                & (boxes[:, 2] < self.mCol - self.mEdge) \
                & (boxes[:, 3] < self.mRow - self.mEdge) 
    + 面积不能太小
    + 识别概率要大于0.6
  
  ## 识别概率为什么用的是bboxes的第五列  probs = bboxes[:,4]  #？？？ 

# G2O的编译指令
cmake -DEigen3_DIR="/home/robotlab/thirdparty/for_dspslam/eigen/install/share/eigen3/cmake" ..

# BDOW2的编译指令
cmake -DOpenCV_DIR="/home/robotlab/work/DSP-SLAM/Thirdparty/opencv/build" ..

# commit 记账
  ## 整合1: Frame语义提取中包含了bbox
  + Frame语义提取中包含了bbox
  + 修改Detection2D函数，实现同时识别多个物体，且能获取语义标签label，根据self.instances = [instance]，可知返回的detections为一个数组，其中每一个py_det为一个detect整合包（包含mask、bbox、label）
  + KeyFrame中加入ORB_SLAM2::Measurements meas， 可以用于存储bbox
  + 下一步，准备利用Tracking::InferObjectsWithSemanticPrior  生成Prior椭球体。 
  + Vector4d(0.00667181,  -0.015002,   0.999865,  0.0274608)
 



     mpOptimizer->SetGroundPlane(groundplane.param);
     