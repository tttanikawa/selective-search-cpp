#include "selective_search.hpp"
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    std::string fileName = "../deer.jpg";
    cv::Mat     img      = cv::imread(fileName, cv::IMREAD_COLOR);

    // selective search
    auto proposals = ss::selectiveSearch(img, 500, 0.8, 50, 20000, 100000, 2.5);
    // do something...

    for (auto&& rect : proposals) {
        cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 3, 8);
    }
    cv::imwrite("./result.jpg", img);
    cv::imshow("result", img);
    cv::waitKey(0);
    return 0;
}