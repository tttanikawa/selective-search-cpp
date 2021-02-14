# C++ Selective Search
![C/C++ CI](https://github.com/ukiyoyo/selective-search-cpp/workflows/C/C++%20CI/badge.svg)

This is a simple, small C++ implementation of Selective Search [1, 2] that can be easily integrated into your projects.

For initial segmentation, this implementation uses Efficient Graph-Based Image Segmentation [3].

![example](example.jpg)

## Dependencies

- C++11 features
- OpenCV (tested version: 4.0)

## Usage

You can test this implementation as:

```sh
% make
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
