# cv2-helper

`cv2-helper` is a Python library designed to make using OpenCV friendlier, faster, and more intuitive. It is especially built for developers, researchers, and students who need to run behavioral tests, debug pipelines, and visualize results in computer vision projects without writing repetitive setup code.

In addition to simplifying native functions, it includes new functionalities and advanced image processing algorithms fully compatible with OpenCV data structures (`numpy.ndarray`).

## Key Features

* **Simplified Visualization:** Forget about combining `cv2.imshow`, `cv2.waitKey`, and `cv2.destroyAllWindows` constantly; visualize images in a single line.

* **Quick Behavioral Testing:** Ideal tools for agile prototyping and visual debugging of image processing pipelines.

* **Extended Algorithms & Utilities:** Additional functions compatible with OpenCV geared towards common visual analysis tasks.

* **Native Integration:** 100% compatible with NumPy arrays and standard OpenCV (`cv2`) functions.

## Installation

You can install `cv2-helper` directly from PyPI using `pip`:

```bash
pip install cv2-helper
```

## Quick Start

Here is a basic example of how to use the `imshow` visualization function to easily display an image:

```python
import cv2
from cv2_helper import imshow

# Load an image using OpenCV
image = cv2.imread("sample_image.jpg")

# Display the image easily with cv2-helper
imshow(image, title="Test Result")
```

## Requirements

* Python $\geq 3.8$
* OpenCV Python (`opencv-python $\geq 4.0.0$`)
* NumPy

## Author

Created and maintained by **Pedro Mayorga**

📧 Contact email: [ppmayorga80@gmail.com](mailto:ppmayorga80@gmail.com)

## License

This project is distributed under the [MIT](LICENSE) license.