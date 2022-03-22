# DenMap: Single Crystal Feature Extraction and Analysis Tool
DenMap is a tool created to auto-detect and analyse dendrite cores in Single Crystal microstructure images. The dendritic core centres are detected using the OpenCV implementation of [template matching](https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html) with the [NCC](https://docs.opencv.org/3.4/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695daf9c3ab9296f597ea71f056399a5831da) (Normalised Cross Correlation) method.

The step-by-step process of detecting cores is as follows:

1. Perform FFT filtering on the loaded image, to select the cores and remove any contrast variations in the image.
2. Binarise the filtered image, selecting the dendritic core centres.
3. The binarised image undergoes OpenCV [contouring](https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a), selecting each core separately.
4. The contours are used to detect the rotation angle of the dendrite cores, and the contours are rotated appropriately.
5. The template for NCC is created by identifying the size of the dendrite cores from the rotated contours.
6. NCC template matching and thresholding is performed, likely dendrite core canditates are detected.
7. Statistical analysis of the dendritic core positions is performed; points that are statistically too close are removed.

Denmap implements all these steps in the auto-process function and, assuming all goes well, the points are detected with a ~95% accuracy. 

In the case where the auto-process is unable to find templates or has poor performance on an image, try the following:

* Check the binary image (View -> Binary Image). Check to see if the dendrite cores are coloured in black but also disconnected from other cores. \
If they are not, select the filtered image (View -> Filtered Image) and then perform the binarisation manually (Process -> Binary Threshold). Once you are satisfied, reperform the auto process (Process -> Auto Process) and it should return with different results.
* If you are unable to binarise the image effectively, repeform the FFT bandpass.\
First, select the original image (View -> Original Image) and click on FFT bandpass (Process -> FFT Bandpass).\
The minimum value should correspond to the pixel size of the dendritic core; if the setting is too high, the image will be very blurry and if the setting is too low it will keep too much detail in the image. Both cases make it difficult to binarise effectively.\
The maximum value corresponds to large structures in the images, such as the background; it is usually find to keep this at a somewhat large value, by default it is set to 100 pixels.
* In the event that the binary image looks satisfactory but the performance of the detected cores is small, try the NCC Thresholding (Process -> NCC Threshold). \
If the theshold is too low, many incorrect core candidates are highlighted and if the theshold is too high, too many core canditates are discared. This value must be tuned accurately.

In the event that you cannot highlight all cores, there is an added function to manually add points; By right-clicking and selecting "Add Point", you can manually add a point.

Denmap can be started by running main.py from this folder
```
python main.py
```

# Package requirements

* Numpy
* Scipy
* Matplotlib
* Scikit-image
* Reikna
    *Note that PyOpenCL and PyCUDA need to be installed for Reikna to be able to use CUDA/OpenCL
* Imutils
* OpenCV (cv2)
* Naturalneighbor ([Github](https://github.com/innolitics/natural-neighbor-interpolation)/[PyPi](https://pypi.org/project/naturalneighbor/))
* Concave_hull ([Github](https://github.com/Geodan/concave-hull))

You must have the relevant compiler ([Windows](https://wiki.python.org/moin/WindowsCompilers)) installed to install Naturalneighbor and Concave_hull. For Windows with Python 3.5+, I recommend Visual C++ v14.0+/Visual Studio 2017+.

Wheels for naturalneighbor and concave_hull that have been compiled using VS Studio 2022 and Python 3.9 are included in wheels directory. These can only be used with Python 3.9.

* Naturalneighbor: If you have a supported compiler installed, this can easily be installed through PyPi using:
    ```cmd
    pip install naturalneighbor
    ```
    In the event that naturalneighbor cannot be imported due to a cnaturalneighbor error, you must create a cnaturalneighbor directory in your ..\Python\Lib\site-packages folder and copy the cnaturalneighbor pyd file (also located in site-packages) into the directory.

* concave_hull: This has not been published on PyPi, so a git clone of the repository must be done. Additionally, it requires [FLANN](https://github.com/flann-lib/flann) be installed along with the [lz4](https://github.com/lz4/lz4) library. Recommended Windows instructions are below:
    1. Install vcpkg using the instructions found on the [lz4](https://github.com/lz4/lz4) github.

        Open command prompt, change your directory to C: (or any other convenient path) and type:
        ```cmd
        git clone https://github.com/Microsoft/vcpkg.git
        cd vcpkg
        bootstrap-vcpkg.bat
        vcpkg integrate install
        ```
    2. Install FLANN and lz4 using vcpkg. Make sure to install the static versions, otherwise your package of concave_hull must have lz4.dll in the same folder.
        ```cmd
        vcpkg install lz4:x64-windows-static
        vcpkg install flann:x64-windows-static
        ```
    3. Set the environmental variable FLANN_DIR to C:\\vcpkg\\installed\\x64-windows-static (change your path to vcpkg accordingly).

        This can be done using command prompt:
        ```
        set FLANN_DIR=C:\vcpkg\installed\x64-windows-static
        ```
    4. You can now clone the concave_hull repository to an appropriate directory. Before installing, you must change an entry in ..\\concave-hull\\setup\\flann_info.py

        Find line 136 with the following text:
        ```python
        'libraries': ['flann', 'flann_cpp']
        ```
        and change it to
        ```python
        'libraries': ['flann', 'flann_cpp', 'lz4']
        ```
        Otherwise it will throw an unreferences LZ4 error during compilation.
    5. Navigate to the cloned directory of concave-hull and run:
        ```cmd
        pip install .
        ```
        If all goes well, it should compile and install without issue. You may have to install the package 'wheel'.

# References

* For feature extraction: Nenchev, B., Strickland, J., Tassenberg, K., Perry, S., Gill, S. and Dong, H., 2020. Automatic Recognition of Dendritic Solidification Structures: DenMap. Journal of Imaging, 6(4), p.19.

* For GUI/automation: Tassenberg, K., Nenchev, B., Strickland, J., Perry, S. and Weston, D., 2020. DenMap Single Crystal Solidification Structure Feature Extraction: Automation and Application, Materials Characterization.

* For characterisation: Strickland, J., Nenchev, B., Perry, S., Tassenberg, K., Gill, S., Panwisawas, C., Dong, H., D'Souza, N. and Irwin, S., 2020. On the nature of hexagonality within the solidification structure of single crystal alloys: Mechanisms and applications. Acta Materialia, 200, pp.417-431.
