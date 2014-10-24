#include <future>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <sensor_msgs/image_encodings.h>

#include "camera_tools/stereo_checker.h"

namespace camera_tools {

StereoChecker::StereoChecker(const ros::NodeHandle& nh,
                             const ros::NodeHandle& pnh)
    : nh_(nh),
      pnh_(pnh),
      it_(nh),
      cfg_server_(nh_),
      detector_(cv::FeatureDetector::create("BRISK")),
      extractor_(cv::DescriptorExtractor::create("BRISK")),
      matcher_(cv::DescriptorMatcher::create("BruteForce-Hamming")) {
  bool approx;
  pnh_.param<bool>("approximate_sync", approx, false);
  int queue_size;
  pnh_.param("queue_size", queue_size, 5);
  SubscribeStereoTopics(approx, queue_size, "image_rect", "camera_info", "raw");
  cfg_server_.setCallback(boost::bind(&StereoChecker::ConfigCb, this, _1, _2));
}

void StereoChecker::SubscribeStereoTopics(const bool approx,
                                          const int queue_size,
                                          const std::string& image_topic,
                                          const std::string& cinfo_topic,
                                          const std::string& transport) {
  image_transport::TransportHints hints(transport, ros::TransportHints(), nh_);

  // Set up camera callback
  if (approx) {
    approximate_sync_.reset(new ApproximateSync(ApproximatePolicy(queue_size),
                                                sub_l_image_, sub_l_cinfo_,
                                                sub_r_image_, sub_r_cinfo_));
    approximate_sync_->registerCallback(
        boost::bind(&StereoChecker::CameraCb, this, _1, _2, _3, _4));
  } else {
    exact_sync_.reset(new ExactSync(ExactPolicy(queue_size), sub_l_image_,
                                    sub_l_cinfo_, sub_r_image_, sub_r_cinfo_));
    exact_sync_->registerCallback(
        boost::bind(&StereoChecker::CameraCb, this, _1, _2, _3, _4));
  }

  // Subscribe to stereo topics
  using namespace ros::names;
  std::string left("left");
  std::string right("right");
  sub_l_image_.subscribe(it_, append(left, image_topic), 1, hints);
  sub_l_cinfo_.subscribe(nh_, append(left, cinfo_topic), 1);
  sub_r_image_.subscribe(it_, append(right, image_topic), 1, hints);
  sub_r_cinfo_.subscribe(nh_, append(right, cinfo_topic), 1);
}

void StereoChecker::ConfigCb(StereoChecker::Config& config, int level) {
  if (level < 0) {
    ROS_INFO("%s: %s", pnh_.getNamespace().c_str(),
             "Initializing reconfigure server");
  }
  config_ = config;
}

void StereoChecker::CameraCb(const ImageConstPtr& l_image_msg,
                             const CameraInfoConstPtr& l_cinfo_msg,
                             const ImageConstPtr& r_image_msg,
                             const CameraInfoConstPtr& r_cinfo_msg) {
  model_.fromCameraInfo(l_cinfo_msg, r_cinfo_msg);
  // Create cv::Mat views onto all buffers
  const cv::Mat_<uint8_t> l_image =
      cv_bridge::toCvShare(l_image_msg, sensor_msgs::image_encodings::MONO8)
          ->image;
  const cv::Mat_<uint8_t> r_image =
      cv_bridge::toCvShare(r_image_msg, sensor_msgs::image_encodings::MONO8)
          ->image;

  // Detect, extract and match features
  std::future<KeypointAndDescriptor> l_future =
      std::async(&StereoChecker::DetectAndExtractFeatures, this, l_image);
  KeypointAndDescriptor l_return = l_future.get();
  std::future<KeypointAndDescriptor> r_future =
      std::async(&StereoChecker::DetectAndExtractFeatures, this, r_image);
  KeypointAndDescriptor r_return = r_future.get();

  std::vector<cv::Point2f> l_pixels, r_pixels;
  std::vector<cv::DMatch> good_matches;
  MatchFeatures(l_return, r_return, good_matches, l_pixels, r_pixels);

  // Calculate match statistics
  PrintMatchQuality(l_pixels, r_pixels);

  // Display
  cv::Mat disp;
  cv::addWeighted(l_image, 0.5, r_image, 0.5, 0.0, disp);
  cv::cvtColor(disp, disp, CV_GRAY2BGR);
  DrawMatches(l_pixels, r_pixels, disp, 2, 2);
  cv::imshow("disp", disp);
  cv::waitKey(1);
}

StereoChecker::KeypointAndDescriptor StereoChecker::DetectAndExtractFeatures(
    const cv::Mat& image) {
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  detector_->detect(image, keypoints);
  extractor_->compute(image, keypoints, descriptors);
  return {keypoints, descriptors};
}

template <typename T>
void StereoChecker::MatchFeatures(const KeypointAndDescriptor& keypts_descs1,
                                  const KeypointAndDescriptor& keypts_descs2,
                                  std::vector<cv::DMatch>& good_matches,
                                  std::vector<cv::Point_<T>>& pixels1,
                                  std::vector<cv::Point_<T>>& pixels2) {

  const std::vector<cv::KeyPoint>& keypoints1 = keypts_descs1.first;
  const std::vector<cv::KeyPoint>& keypoints2 = keypts_descs2.first;
  const cv::Mat& desc1 = keypts_descs1.second;
  const cv::Mat& desc2 = keypts_descs2.second;

  //  std::cout << desc1.rows << " " << desc2.rows << std::endl;
  //  ROS_ASSERT_MSG(desc1.rows == desc2.rows, "Descriptor size mismatch");
  std::vector<cv::DMatch> matches;
  matcher_->match(desc1, desc2, matches);
  auto min_dist = std::numeric_limits<decltype(cv::DMatch::distance)>::max();
  // Calculate min distance
  for (const cv::DMatch& match : matches) {
    min_dist = std::min(min_dist, match.distance);
  }
  // Refine matches
  auto dist_thresh = config_.dist_thresh * min_dist;
  for (const cv::DMatch& match : matches) {
    if (match.distance <= dist_thresh) {
      int id1 = match.queryIdx;
      int id2 = match.trainIdx;
      auto& pts1 = keypoints1[id1];
      auto& pts2 = keypoints2[id2];
      if (std::abs(pts1.pt.y - pts2.pt.y) < config_.delta_v_thresh) {
        good_matches.push_back(match);
        pixels1.emplace_back(pts1.pt.x, pts1.pt.y);
        pixels2.emplace_back(pts2.pt.x, pts2.pt.y);
      }
    }
  }
}

template <typename T>
void DrawMatches(const std::vector<cv::Point_<T>>& points1,
                 const std::vector<cv::Point_<T>>& points2, cv::Mat& image,
                 int radius, int thickness) {
  size_t num_pts = points1.size();
  for (size_t i = 0; i < num_pts; ++i) {
    cv::circle(image, points1[i], radius, cv::Scalar(255, 0, 0), thickness);
    cv::circle(image, points2[i], radius, cv::Scalar(0, 0, 255), thickness);
    cv::line(image, points1[i], points2[i], cv::Scalar(0, 255, 0), thickness);
  }
}

template <typename T>
void PrintMatchQuality(const std::vector<cv::Point_<T>>& points1,
                       const std::vector<cv::Point_<T>>& points2) {
  double sum_delta_v{0};
  for (size_t i = 0; i < points1.size(); ++i) {
    sum_delta_v += std::abs(points1[i].y - points2[i].y);
  }
  double ave_delta_v = sum_delta_v / points1.size();
  ROS_INFO_STREAM_THROTTLE(
      1, "Matches: " << points1.size()
                     << " , average delta_v: " << ave_delta_v);
}

}  // namespace bluefox2

int main(int argc, char** argv) {
  ros::init(argc, argv, "stereo_checker");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  try {
    camera_tools::StereoChecker stereo_checker(nh, pnh);
    ros::spin();
  }
  catch (const std::exception& e) {
    ROS_ERROR("%s: %s", nh.getNamespace().c_str(), e.what());
  }
}
