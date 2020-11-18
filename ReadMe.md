# SurroundVision2BEV

## Usage
1. Use `calibrate_distort.py` for getting cameras' K & D;
2. Use `get_image.py` for testing K & D and prepare the images for next step; 
3. Use `calibrate_perspective.py` (x, y, z, roll, pitch, yaw)
   or `calibrate_homography.py` (4 points)
   for adjusting the homogeneous matrix.

## Requirements
```
opencv-python
numpy
easydict
```
