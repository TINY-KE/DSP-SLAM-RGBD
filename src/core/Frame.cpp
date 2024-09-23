#include <include/core/Frame.h>

namespace EllipsoidSLAM
{
    int Frame::total_frame=0;

    void Frame::Clear()
    {
        total_frame = 0;
    }

    Eigen::MatrixXd FilterDetection(const Eigen::MatrixXd &bboxMap)
    {
        // 本函数用于过滤YOLO中一个物体出现两次的情况.
        // 即发现大小一样的物体，并且保留概率更大的
        Eigen::MatrixXd bboxMap_filtered;

        int obj_num = bboxMap.rows();
        for(int i=0;i<obj_num;i++)
        {
            Eigen::VectorXd vec = bboxMap.row(i);
            // 做差求和，如果一样则比较取概率更高的

            Eigen::VectorXd bbox = vec.head(5).tail(4);
            double prob = vec[6];

            int exist_num = bboxMap_filtered.rows();

            bool bFindSame = false;
            for(int n=0;n<exist_num;n++){
                Eigen::VectorXd vec_exist = bboxMap_filtered.row(n);
                Eigen::VectorXd bbox_exist = vec_exist.head(5).tail(4);
                double prob_exist = vec_exist[6];

                // DEBUG
                if(bbox_exist.rows() != bbox.rows())
                {
                    std::cout << "Please check size: " << std::endl;
                    std::cout << "bbox_exist : " << bbox_exist.transpose() << std::endl;
                    std::cout << "bbox : " << bbox.transpose() << std::endl;
                }

                double diff = (bbox_exist - bbox).norm();
                if(diff < 0.1)
                {
                    // find a same bbox
                    if(prob > prob_exist)   // 保留更高的
                    {
                        // 替换该行
                        bboxMap_filtered.row(n) = vec;
                    }
                    bFindSame = true;   // 无论是否替换，都结束了
                }

                if(bFindSame) break;
            }
            if(!bFindSame)
            {
                // 添加进去
                addVecToMatirx(bboxMap_filtered, vec);
            }
        }

        return bboxMap_filtered;

    }

    Frame::Frame(double timestamp_, const Eigen::VectorXd &pose, const Eigen::MatrixXd &bboxMap, const cv::Mat &imDepth, const cv::Mat &imRGB, bool verbose)
    {
        timestamp = timestamp_;
        rgb_img = imRGB.clone();
        frame_img = imDepth.clone();
        if(!rgb_img.empty())
            cv::cvtColor(rgb_img, gray_img, CV_BGR2GRAY);

        cam_pose_Twc.fromVector(pose.tail(7));
        cam_pose_Tcw = cam_pose_Twc.inverse();

        frame_seq_id = total_frame++;

        mmObservations = FilterDetection(bboxMap);   // 注意做一波过滤

        mvbOutlier.resize(mmObservations.rows());
        fill(mvbOutlier.begin(), mvbOutlier.end(), false);  

        if(verbose){
            std::cout << "--------> New Frame : " << frame_seq_id << " Timestamp: " << std::to_string(timestamp) << std::endl;
            std::cout << "[Frame.cpp] mmObservations : " << std::endl << mmObservations << std::endl << std::endl;
        }

        mbHaveLocalObject = false;
        mbSetRelation = false;
    }


}
