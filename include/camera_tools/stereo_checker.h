#ifndef CAMERA_TOOLS_STEREO_CHECKER_H_
#define CAMERA_TOOLS_STEREO_CHECKER_H_

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_geometry/stereo_camera_model.h>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>

#include <camera_tools/StereoCheckerDynConfig.h>

namespace camera_tools {

using sensor_msgs::Image;
using sensor_msgs::CameraInfo;
using sensor_msgs::ImageConstPtr;
using sensor_msgs::CameraInfoConstPtr;
using message_filters::sync_policies::ExactTime;
using message_filters::sync_policies::ApproximateTime;

class StereoChecker {

 public:
  typedef camera_tools::StereoCheckerDynConfig Config;
  typedef std::pair<std::vector<cv::KeyPoint>, cv::Mat> KeypointAndDescriptor;

  StereoChecker(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  StereoChecker(const StereoChecker&) = delete;
  StereoChecker& operator=(const StereoChecker&) = delete;

  void CameraCb(const ImageConstPtr& l_image_msg,
                const CameraInfoConstPtr& l_cinfo_msg,
                const ImageConstPtr& r_image_msg,
                const CameraInfoConstPtr& r_cinfo_msg);
  void ConfigCb(StereoChecker::Config& config, int level);

 private:
  void SubscribeStereoTopics(const bool approx, const int queue_size,
                             const std::string& image_topic,
                             const std::string& cinfo_topic,
                             const std::string& transport);

  KeypointAndDescriptor DetectAndExtractFeatures(const cv::Mat& image);

  template <typename T>
  void MatchFeatures(const KeypointAndDescriptor& keypts_descs1,
                     const KeypointAndDescriptor& keypts_descs2,
                     std::vector<cv::DMatch>& good_matches,
                     std::vector<cv::Point_<T>>& pixels1,
                     std::vector<cv::Point_<T>>& pixels2);

  ros::NodeHandle nh_, pnh_;
  image_transport::ImageTransport it_;
  image_transport::SubscriberFilter sub_l_image_, sub_r_image_;
  message_filters::Subscriber<CameraInfo> sub_l_cinfo_, sub_r_cinfo_;
  typedef ExactTime<Image, CameraInfo, Image, CameraInfo> ExactPolicy;
  typedef ApproximateTime<Image, CameraInfo, Image, CameraInfo>
      ApproximatePolicy;
  typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
  typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;
  dynamic_reconfigure::Server<Config> cfg_server_;
  boost::shared_ptr<ExactSync> exact_sync_;
  boost::shared_ptr<ApproximateSync> approximate_sync_;
  image_geometry::StereoCameraModel model_;
  cv::Ptr<cv::FeatureDetector> detector_;
  cv::Ptr<cv::DescriptorExtractor> extractor_;
  cv::Ptr<cv::DescriptorMatcher> matcher_;
  Config config_;
};

template <typename T>
void DrawMatches(const std::vector<cv::Point_<T>>& points1,
                 const std::vector<cv::Point_<T>>& points2, cv::Mat& image,
                 int raidus, int thickness);

template <typename T>
void PrintMatchQuality(const std::vector<cv::Point_<T>>& points1,
                       const std::vector<cv::Point_<T>>& points2);

}  // namespace camera_tools

#endif  // CAMERA_TOOLS_STEREO_CHECKER_H_
