#pragma once


#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>


#define CV_VERSION_STR CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#ifdef _DEBUG
#define CV_EXT_STR "d.lib"
#else
#define CV_EXT_STR ".lib"
#endif


#pragma comment(lib, "opencv_world"	CV_VERSION_STR CV_EXT_STR)


namespace std
{
	template<>
	class hash<std::pair<int, int>>
	{
	public:
		std::size_t operator()( const std::pair<int, int> &x ) const
		{
			return hash<int>()( x.first ) ^ hash<int>()( x.second );
		}
	};
}



namespace ss
{

	inline double square( double a )
	{
		return a*a;
	}


	inline double diff( const cv::Mat &img, int x1, int y1, int x2, int y2 )
	{
		return sqrt( square( img.at<cv::Vec3f>( y1, x1 )[0] - img.at<cv::Vec3f>( y2, x2 )[0] ) +
			square( img.at<cv::Vec3f>( y1, x1 )[1] - img.at<cv::Vec3f>( y2, x2 )[1] ) +
			square( img.at<cv::Vec3f>( y1, x1 )[2] - img.at<cv::Vec3f>( y2, x2 )[2] ) );
	}


	struct UniverseElement
	{
		int rank;
		int p;
		int size;

		UniverseElement() : rank( 0 ), size( 1 ), p( 0 ) {}
		UniverseElement( int rank, int size, int p ) : rank( rank ), size( size ), p( p ) {}

		bool operator ==(const UniverseElement& other) const
		{
			return rank == other.rank && p == other.p && size == other.size;
		}
	};


	class Universe
	{
	private:
		std::vector<UniverseElement> elements;
		int num;

	public:
		Universe( int num ) : num( num )
		{
			elements.reserve( num );

			for ( int i = 0; i < num; i++ )
			{
				elements.emplace_back( 0, 1, i );
			}
		}

		
		~Universe() {}

		int findFast(int x)
		{
			return elements[x].p;
		}

		int find( int x )
		{
			int y = x;
			while ( y != elements[y].p )
			{
				y = elements[y].p;
			}
			elements[x].p = y;

			return y;
		}

		void join( int x, int y )
		{
			if ( elements[x].rank > elements[y].rank )
			{
				elements[y].p = x;
				elements[x].size += elements[y].size;
			}
			else
			{
				elements[x].p = y;
				elements[y].size += elements[x].size;
				if ( elements[x].rank == elements[y].rank )
				{
					elements[y].rank++;
				}
			}
			num--;
		}

		int size( int x ) const { return elements[x].size; }
		int numSets() const { return num; }
	};


	struct edge
	{
		int a;
		int b;
		double w;
	};


	bool operator<( const edge &a, const edge &b )
	{
		return a.w < b.w;
	}


	inline double calThreshold( int size, double scale )
	{
		return scale / size;
	}


	std::shared_ptr<Universe> segmentGraph( int numVertices, int numEdges, std::vector<edge> &edges, double scale )
	{
		std::sort( edges.begin(), edges.end() );

		auto universe = std::make_shared<Universe>( numVertices );

		std::vector<double> threshold( numVertices, scale );

		for ( auto &pedge : edges )
		{
			int a = universe->find( pedge.a );
			int b = universe->find( pedge.b );

			if ( a != b )
			{
				if ( ( pedge.w <= threshold[a] ) && ( pedge.w <= threshold[b] ) )
				{
					universe->join( a, b );
					a = universe->find( a );
					threshold[a] = pedge.w + calThreshold( universe->size( a ), scale );
				}
			}
		}

		return universe;
	}


	// image segmentation using "Efficient Graph-Based Image Segmentation"
	std::shared_ptr<Universe> segmentation( const cv::Mat &img, double scale, double sigma, int minSize )
	{
		const int width = img.cols;
		const int height = img.rows;

		cv::Mat imgF;
		img.convertTo( imgF, CV_32FC3 );

		cv::Mat blurred;
		cv::GaussianBlur( imgF, blurred, cv::Size( 5, 5 ), sigma );

		std::vector<edge> edges( width*height * 4 );

		int num = 0;
		for ( int y = 0; y < height; y++ )
		{
			for ( int x = 0; x < width; x++ )
			{
				if ( x < width - 1 )
				{
					edges[num].a = y * width + x;
					edges[num].b = y * width + ( x + 1 );
					edges[num].w = diff( blurred, x, y, x + 1, y );
					num++;
				}

				if ( y < height - 1 )
				{
					edges[num].a = y * width + x;
					edges[num].b = ( y + 1 ) * width + x;
					edges[num].w = diff( blurred, x, y, x, y + 1 );
					num++;
				}

				if ( ( x < width - 1 ) && ( y < height - 1 ) )
				{
					edges[num].a = y * width + x;
					edges[num].b = ( y + 1 ) * width + ( x + 1 );
					edges[num].w = diff( blurred, x, y, x + 1, y + 1 );
					num++;
				}

				if ( ( x < width - 1 ) && ( y > 0 ) )
				{
					edges[num].a = y * width + x;
					edges[num].b = ( y - 1 ) * width + ( x + 1 );
					edges[num].w = diff( blurred, x, y, x + 1, y - 1 );
					num++;
				}
			}
		}

		auto universe = segmentGraph( width*height, num, edges, scale );


		for ( int i = 0; i < num; i++ )
		{
			int a = universe->find( edges[i].a );
			int b = universe->find( edges[i].b );
			if ( ( a != b ) && ( ( universe->size( a ) < minSize ) || ( universe->size( b ) < minSize ) ) )
			{
				universe->join( a, b );
			}
		}

		return universe;
	}


	void visualize( const cv::Mat &img, std::shared_ptr<Universe> universe )
	{
		const int height = img.rows;
		const int width = img.cols;
		std::vector<cv::Vec3b> colors;

		cv::Mat segmentated( height, width, CV_8UC3 );

		std::random_device rnd;
		std::mt19937 mt( rnd() );
		std::uniform_int_distribution<> rand256( 0, 255 );

		for ( int i = 0; i < height*width; i++ )
		{
			cv::Vec3b color( rand256( mt ), rand256( mt ), rand256( mt ) );
			colors.push_back( color );
		}

		for ( int y = 0; y < height; y++ )
		{
			for ( int x = 0; x < width; x++ )
			{
				segmentated.at<cv::Vec3b>( y, x ) = colors[universe->findFast( y*width + x )];
			}
		}

		cv::imshow( "Initial Segmentation Result", segmentated );
		cv::waitKey( 1 );
	}


	struct Region
	{
		int size;
		cv::Rect rect;
		std::vector<int> labels;
		std::vector<float> colourHist;
		std::vector<float> textureHist;
		std::vector<cv::Vec2i> points;

		Region() {}

		Region( const cv::Rect &rect, int label ) : rect( rect )
		{
			labels.push_back( label );
		}

		Region(
			const cv::Rect &rect, int size,
			const std::vector<float> &&colourHist,
			const std::vector<float> &&textureHist,
			const std::vector<int> &&labels
			)
			: rect( rect ), size( size ), colourHist( std::move( colourHist ) ), textureHist( std::move( textureHist ) ), labels( std::move( labels ) )
		{}

		Region& operator=( const Region& region ) = default;

		Region& operator=( Region&& region ) noexcept
		{
			if ( this != &region )
			{
				this->size = region.size;
				this->rect = region.rect;
				this->labels = std::move( region.labels );
				this->colourHist = std::move( region.colourHist );
				this->textureHist = std::move( region.textureHist );
			}

			return *this;
		}

		Region( Region&& region ) noexcept
		{
			*this = std::move( region );
		}
	};


	std::shared_ptr<Universe> generateSegments( const cv::Mat &img, double scale, double sigma, int minSize )
	{
		auto universe = segmentation( img, scale, sigma, minSize );

		visualize( img, universe );

		return universe;
	}


	double calcSimOfColour( const Region &r1, const Region &r2 )
	{
		assert( r1.colourHist.size() == r2.colourHist.size() );

		float sum = 0.0;

		for ( auto i1 = r1.colourHist.cbegin(), i2 = r2.colourHist.cbegin(); i1 != r1.colourHist.cend(); i1++, i2++ )
		{
			sum += std::min( *i1, *i2 );
		}

		return sum;
	}


	double calcSimOfTexture( const Region &r1, const Region &r2 )
	{
		assert( r1.colourHist.size() == r2.colourHist.size() );

		double sum = 0.0;

		for ( auto i1 = r1.textureHist.cbegin(), i2 = r2.textureHist.cbegin(); i1 != r1.textureHist.cend(); i1++, i2++ )
		{
			sum += std::min( *i1, *i2 );
		}

		return sum;
	}


	inline double calcSimOfSize( const Region &r1, const Region &r2, int imSize )
	{
		return ( 1.0 - ( double )( r1.size + r2.size ) / imSize );
	}


	inline double calcSimOfRect( const Region &r1, const Region &r2, int imSize )
	{
		return ( 1.0 - ( double )( ( r1.rect | r2.rect ).area() - r1.size - r2.size ) / imSize );
	}


	inline double calcSimilarity( const Region &r1, const Region &r2, int imSize )
	{
		return ( calcSimOfColour( r1, r2 ) + calcSimOfTexture( r1, r2 ) + calcSimOfSize( r1, r2, imSize ) + calcSimOfRect( r1, r2, imSize ) );
	}


	void calcColourHist( const cv::Mat &img, std::shared_ptr<Universe> universe, int label, Region& region)
	{
		std::array<std::vector<unsigned char>, 3> hsv;

		for ( auto &e : hsv )
		{
			e.reserve(region.points.size());
		}

		for (cv::Vec2i point : region.points)
		{
			for (int channel = 0; channel < 3; channel++)
			{
				hsv[channel].push_back(img.at<cv::Vec3b>(point[0], point[1])[channel]);
			}
		}

		int channels[] = { 0 };
		const int bins = 25;
		int histSize[] = { bins };
		float range[] = { 0, 256 };
		const float *ranges[] = { range };


		for ( int channel = 0; channel < 3; channel++ )
		{
			cv::Mat hist;

			cv::Mat input( hsv[channel] );

			cv::calcHist( &input, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false );

			cv::normalize( hist, hist, 1.0, 0.0, cv::NORM_L1 );

			std::vector<float> histogram;
			hist.copyTo( histogram );

			if (region.colourHist.empty() )
			{
				region.colourHist = std::move( histogram );
			}
			else
			{
				std::copy( histogram.begin(), histogram.end(), std::back_inserter(region.colourHist) );
			}
		}
	}


	cv::Mat calcTextureGradient( const cv::Mat &img )
	{
		cv::Mat sobelX, sobelY;

		cv::Sobel( img, sobelX, CV_32F, 1, 0 );
		cv::Sobel( img, sobelY, CV_32F, 0, 1 );

		cv::Mat magnitude, angle;
		cv::cartToPolar( sobelX, sobelY, magnitude, angle, true );

		return angle;
	}


	void calcTextureHist( const cv::Mat &img, const cv::Mat &gradient, std::shared_ptr<Universe> universe, int label, Region& region )
	{
		const int orientations = 8;

		std::array<std::array<std::vector<unsigned char>, orientations>, 3> intensity;

		for ( auto &e : intensity )
		{
			for ( auto &ee : e )
			{
				ee.reserve(region.points.size());
			}
		}

		for (cv::Vec2i point : region.points)
		{
			for (int channel = 0; channel < 3; channel++)
			{
				int angle = (int)(gradient.at<cv::Vec3f>(point[0], point[1])[channel] / 22.5) % orientations;
				intensity[channel][angle].push_back(img.at<cv::Vec3b>(point[0], point[1])[channel]);
			}
		}


		int channels[] = { 0 };
		const int bins = 10;
		int histSize[] = { bins };
		float range[] = { 0, 256 };
		const float *ranges[] = { range };

		for ( int channel = 0; channel < 3; channel++ )
		{
			for ( int angle = 0; angle < orientations; angle++ )
			{
				cv::Mat hist;

				cv::Mat input( intensity[channel][angle] );

				cv::calcHist( &input, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false );

				cv::normalize( hist, hist, 1.0, 0.0, cv::NORM_L1 );

				std::vector<float> histogram;
				hist.copyTo( histogram );

				if (region.textureHist.empty() )
				{
					region.textureHist = std::move( histogram );
				}
				else
				{
					std::copy( histogram.begin(), histogram.end(), std::back_inserter(region.textureHist) );
				}

			}
		}
	}




	std::map<int, Region> extractRegions( const cv::Mat &img, std::shared_ptr<Universe> universe )
	{
		std::map<int, Region> R;
		

		for ( int y = 0; y < img.rows; y++ )
		{
			for ( int x = 0; x < img.cols; x++ )
			{
				int label = universe->findFast( y*img.cols + x );

				if ( R.find( label ) == R.end() )
				{
					R[label] = Region( cv::Rect( 100000, 100000, 0, 0 ), label );
				}

				Region& region = R[label];

				if (region.rect.x > x )
				{
					region.rect.x = x;
				}

				if (region.rect.y > y )
				{
					region.rect.y = y;
				}

				if (region.rect.br().x < x )
				{
					region.rect.width = x - region.rect.x + 1;
				}

				if (region.rect.br().y < y )
				{
					region.rect.height = y - region.rect.y + 1;
				}
				region.points.push_back(cv::Vec2i(y, x));
			}
		}


		cv::Mat gradient = calcTextureGradient( img );

		cv::Mat hsv;
		cv::cvtColor( img, hsv, cv::COLOR_BGR2HSV );

		for ( auto &labelRegion : R )
		{
			labelRegion.second.size = labelRegion.second.points.size();
			calcColourHist( hsv, universe, labelRegion.first, labelRegion.second);
			calcTextureHist( img, gradient, universe, labelRegion.first, labelRegion.second);
		}

		return R;
	}


	inline bool isIntersecting( const Region &a, const Region &b )
	{
		return ( ( a.rect & b.rect ).area() != 0 );
	}


	using LabelRegion = std::pair<int, Region>;
	using Neighbour = std::pair<int, int>;


	std::vector<Neighbour> extractNeighbours( const std::map<int, Region> &R )
	{
		std::vector<Neighbour> neighbours;
		neighbours.reserve( R.size()*( R.size() - 1 ) / 2 );

		for ( auto a = R.cbegin(); a != R.cend(); a++ )
		{
			auto tmp = a;
			tmp++;

			for ( auto b = tmp; b != R.cend(); b++ )
			{
				if ( isIntersecting( a->second, b->second ) )
				{
					neighbours.push_back( std::make_pair( std::min( a->first, b->first ), std::max( a->first, b->first ) ) );
				}
			}
		}

		return neighbours;
	}


	std::vector<float> merge( const std::vector<float> &a, const std::vector<float> &b, int asize, int bsize )
	{
		std::vector<float> newVector;
		newVector.reserve( a.size() );

		for ( auto ai = a.begin(), bi = b.begin(); ai != a.end(); ai++, bi++ )
		{
			newVector.push_back( ( ( *ai ) * asize + ( *bi ) * bsize ) / ( asize + bsize ) );
		}

		return newVector;
	};


	Region mergeRegions( const Region &r1, const Region &r2 )
	{
		assert( r1.colourHist.size() == r2.colourHist.size() );
		assert( r1.textureHist.size() == r2.textureHist.size() );

		int newSize = r1.size + r2.size;

		std::vector<int> newLabels( r1.labels );
		std::copy( r2.labels.begin(), r2.labels.end(), std::back_inserter( newLabels ) );

		return Region( r1.rect | r2.rect,
			newSize,
			std::move( merge( r1.colourHist, r2.colourHist, r1.size, r2.size ) ),
			std::move( merge( r1.textureHist, r2.textureHist, r1.size, r2.size ) ),
			std::move( newLabels )
			);
	}


	std::vector<cv::Rect> selectiveSearch( const cv::Mat &img, double scale = 1.0, double sigma = 0.8, int minSize = 50, int smallest = 1000, int largest = 270000, double distorted = 5.0 )
	{
		assert( img.channels() == 3 );

		auto universe = generateSegments( img, scale, sigma, minSize );
		int imgSize = img.total();

		auto R = extractRegions( img, universe );

		auto neighbours = extractNeighbours( R );

		std::unordered_map<std::pair<int, int>, double> S;

		for ( auto &n : neighbours )
		{
			S[n] = calcSimilarity( R[n.first], R[n.second], imgSize );
		}

		using NeighbourSim = std::pair<std::pair<int, int>, double >;

		while ( !S.empty() )
		{
			auto cmp = []( const NeighbourSim &a, const NeighbourSim &b ) { return a.second < b.second; };

			auto m = std::max_element( S.begin(), S.end(), cmp );

			int i = m->first.first;
			int j = m->first.second;
			auto ij = std::make_pair( i, j );

			int t = R.rbegin()->first + 1;
			R[t] = mergeRegions( R[i], R[j] );

			std::vector<std::pair<int, int>> keyToDelete;

			for ( auto &s : S )
			{
				auto key = s.first;

				if ( ( i == key.first ) || ( i == key.second ) || ( j == key.first ) || ( j == key.second ) )
				{
					keyToDelete.push_back( key );
				}
			}

			for ( auto &key : keyToDelete )
			{
				S.erase( key );

				if ( key == ij )
				{
					continue;
				}

				int n = ( key.first == i || key.first == j ) ? key.second : key.first;
				S[std::make_pair( n, t )] = calcSimilarity( R[n], R[t], imgSize );
			}
		}

		std::vector<cv::Rect> proposals;
		proposals.reserve( R.size() );

		for ( auto &r : R )
		{
			// exclude same rectangle (with different segments)
			if ( std::find( proposals.begin(), proposals.end(), r.second.rect ) != proposals.end() )
			{
				continue;
			}

			// exclude regions that is smaller/larger than assigned size
			if ( r.second.size < smallest || r.second.size > largest )
			{
				continue;
			}

			double w = r.second.rect.width;
			double h = r.second.rect.height;

			// exclude distorted rects
			if ( ( w / h > distorted ) || ( h / w > distorted ) )
			{
				continue;
			}

			proposals.push_back( r.second.rect );
		}

		return proposals;
	}
}
