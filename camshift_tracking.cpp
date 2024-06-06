// =====================================================================
//  Filename:    camshift_tracking.cpp
//  Description: Recognizes regions of text in a given image
//  Usage: ./camshift_tracking
//         or
//         ./camshift_tracking --video test.mov
//
//  Author: Raghunath N (raghumtech@gmail.com)
// =====================================================================

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// global variables to be used
cv::Mat frame;
std::vector<cv::Point> roi_points;
bool input_mode = false;

void get_arguments(int argc, char** argv, std::string& video_path) 
{
    cv::CommandLineParser parser(argc, argv, "{help h||}{@video||}");
    if (parser.has("help")) 
    {
        parser.printMessage();
        return;
    }
    video_path = parser.get<std::string>("@video");
}

static void select_roi(int event, int x, int y, int flags, void* param) 
{
    /**
     * Draw circles at the selection region vertices and show the selected ROI on image
     * :param event: mouse callback event from openCV
     * :param x: x coordinate of pointer
     * :param y: y coordinate of pointer
     */
    if (input_mode && event == cv::EVENT_LBUTTONDOWN && roi_points.size() < 4) 
    {
        roi_points.emplace_back(x, y);
        cv::circle(frame, cv::Point(x, y), 4, cv::Scalar(0, 255, 0), 2);
        cv::imshow("frame", frame);
    }
}

std::pair<cv::Mat, cv::Rect> frame_roi() 
{
    /**
     * Freezes frame on entering insert mode. Upon selecting ROI, converts ROI to HSV color space
     * and calculates its hue histogram
     * :return: roi histogram, roi box (tuple)
     */
    input_mode = true;
    cv::Mat orig_frame = frame.clone();

    // only frame when 4 points are not selected
    while (roi_points.size() < 4) 
    {
        cv::imshow("frame", frame);
        cv::waitKey(0);
    }

    std::vector<cv::Point> roi_points_vec(roi_points);
    cv::Rect roi_box;
    cv::Point top_left, bottom_right;
    int min_sum = INT_MAX, max_sum = 0;
    for (const auto& point : roi_points_vec)
      {
        int sum = point.x + point.y;
        if (sum < min_sum) 
        {
            min_sum = sum;
            top_left = point;
        }
        if (sum > max_sum) 
        {
            max_sum = sum;
            bottom_right = point;
        }
    }
    roi_box = cv::Rect(top_left, bottom_right);

    // convert ROI to HSV color space
    cv::Mat roi = orig_frame(roi_box);
    cv::cvtColor(roi, roi, cv::COLOR_BGR2HSV);

    // calculate histogram for the ROI & normalize it
    cv::Mat roi_hist;
    int histSize = 16;
    float range[] = {0, 180};
    const float* histRange = {range};
    cv::calcHist(&roi, 1, 0, cv::Mat(), roi_hist, 1, &histSize, &histRange);
    cv::normalize(roi_hist, roi_hist, 0, 255, cv::NORM_MINMAX);

    return std::make_pair(roi_hist, roi_box);
}

void apply_camshift(const cv::Rect& roi_box, cv::TermCriteria termination, const cv::Mat& roi_hist) 
{
    /**
     * Applies the camshift algorithm to a back projection of the HSV color space ROI
     * :param roi_box: region of interest box
     * :param termination: termination criteria for the camshift algorithm iterations
     * :param roi_hist: region of interest histogram
     */
    // calculate back projection for the ROI
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::Mat back_projection;
    cv::calcBackProject(&hsv, 1, 0, roi_hist, back_projection, &roi_hist);

    // apply the camshift algorithm to the ROI
    cv::RotatedRect r = cv::CamShift(back_projection, roi_box, termination);
    std::vector<cv::Point> points(4);
    cv::boxPoints(r, points.data());
    cv::polylines(frame, points, true, cv::Scalar(0, 255, 0), 2);
}

int main(int argc, char** argv) 
{
    std::string video_path;
    get_arguments(argc, argv, video_path);

    cv::VideoCapture camera;
    if (video_path.empty()) 
    {
        // start web cam feed
        camera.open(0);
    } 
    else 
    {
        // load video file
        camera.open(video_path);
    }

    cv::namedWindow("frame");
    cv::setMouseCallback("frame", select_roi, nullptr);

    cv::TermCriteria termination(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1);
    cv::Rect roi_box;
    cv::Mat roi_hist;

    // main loop
    while (true) 
    {
        camera >> frame;
        if (frame.empty()) 
        {
            break;
        }
        // apply camshift if ROI is selected
        if (!roi_box.empty()) 
        {
            apply_camshift(roi_box, termination, roi_hist);
        }
        // show the feed/results
        cv::imshow("frame", frame);
        char key = static_cast<char>(cv::waitKey(1));

        // insert mode to select ROI
        if (key == 'i' && roi_points.size() < 4) 
        {
            std::tie(roi_hist, roi_box) = frame_roi();
        }

        // quit if 'q' is pressed
        else if (key == 'q') 
        {
            break;
        }
    }
    // clean up endpoints
    camera.release();
    cv::destroyAllWindows();
    return 0;
}

