#include <opencv2/opencv.hpp>
#include "selective_search.hpp"


int main( int argc, char** argv )
{
	std::string fileName = "../deer.jpg";

	cv::Mat img = cv::imread( fileName, cv::IMREAD_COLOR );

	// selective search
	auto proposals = ss::selectiveSearch( img, 380, 0.8, 100, 600, 25000, 3.0 );
	
	// do something...

	for ( auto &&rect : proposals )
	{
		cv::rectangle( img, rect, cv::Scalar( 0, 255, 0 ), 3, 8 );
	}

	cv::imshow( "result", img );
	cv::waitKey( 0 );

	return 0;
}