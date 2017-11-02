package mainCpp;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Size;

public class ImageProcessCPP {
    private opencv_videoio.VideoCapture capture;
    private Mat staticBackground = new Mat();
    private Mat currentFrame = new Mat();
    private Size gaussianFilterSize = (new Size(3, 3));
    opencv_xfeatures2d.SURF surf;

    public ImageProcessCPP(opencv_videoio.VideoCapture capture) {
        this.capture = capture;

        this.surf = opencv_xfeatures2d.SURF.create();
        this.surf.setUpright(false);
    }

    public Mat getOriginalFrame() {
        if (this.capture.isOpened()) {
            try {
                this.capture.read(currentFrame);
            } catch (Exception e) {
                System.err.println("Exception during the image elaboration: " + e);
            }
        }
        return currentFrame;
    }

    public opencv_core.KeyPointVector getSURFKeyPoint(Mat input, Mat mask) {
        opencv_core.KeyPointVector keyPointVector = new opencv_core.KeyPointVector();
        surf.detect(input, keyPointVector, mask);
        return keyPointVector;
    }

    public Mat getGaussianBlur() {
        Mat blurFrame = new Mat();
        if (!currentFrame.empty()) {
            opencv_imgproc.GaussianBlur(currentFrame, blurFrame, gaussianFilterSize, 0);
        }
        return blurFrame;
    }

    public Mat setStaticBackground() {
        Mat frame = new Mat();
        if (this.capture.isOpened()) {
            try {
                this.capture.read(frame);
                if (!frame.empty()) {
                    staticBackground = frame;
                }
            } catch (Exception e) {
                System.err.println("Exception during the image elaboration: " + e);
            }
        }
        return frame;
    }

    public void setGaussianFilterSize(int size) {
        int validSize = (size % 2) != 0 ? size : size - 1;
        log("Change Gaussian filter size to:" + validSize);
        gaussianFilterSize = new Size(validSize, validSize);
    }

    public void setHessianThreshold(int value) {
        log("Change Hessian Threshold to:" + value);
        surf.setHessianThreshold(value);
    }

    public void setNOctaveLayer(int value) {
        log("Change Hessian Threshold to:" + value);
        surf.setNOctaveLayers(value);
    }


    private void log(Object o) {
        System.out.println(o);
    }
}
