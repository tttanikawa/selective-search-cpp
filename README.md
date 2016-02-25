# C++ Selective Search

This is a C++ implementation of Selective Search [1, 2].

For initial segmentation, this implementation uses Efficient Graph-Based Image Segmentation [3].

![example](example.jpg)

## Dependencies

- Visual Studio 2013 / GCC 4.8 / Clang 3.2 or later
	* This implementation relies on C++11 features.
- OpenCV (2.4.9-)

## Usage

You can test this implementation as:

```sh
% ./test
```

To include it in your project, you just need to:

```cpp
#include "selective_search.hpp"

...

// Get object proposals
auto proposals = ss::selectiveSearch( img, scale, sigma, minSize, smallest, largest, distorted );

```

## License

MIT

## References

[1] J. R. R. Uijlings et al., "Selective Search for Object Recognition", IJCV, 2013

[2] K. E. A. van de Sande et al., "Segmentation As Selective Search for Object Recognition", ICCV, 2011

[3] P. Felzenszwalb et al., "Efficient Graph-Based Image Segmentation", IJCV, 2004