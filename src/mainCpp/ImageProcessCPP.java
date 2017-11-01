package mainCpp;

import algorithms.KernelDensityEstimator;
import algorithms.Vibe;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_video;
import org.bytedeco.javacpp.opencv_videoio;
import org.bytedeco.javacpp.opencv_imgproc;
import org.opencv.imgproc.Imgproc;

public class ImageProcessCPP {
    private opencv_videoio.VideoCapture capture;
    private Mat staticBackground = new Mat();
    private Mat currentFrame = new Mat();
    private opencv_video.BackgroundSubtractorMOG2 pMOG2;
    private opencv_video.BackgroundSubtractorKNN pknn;
    private Size gaussianFilterSize = (new Size(3, 3));
    KernelDensityEstimator kde;
    Vibe vibe;

    public ImageProcessCPP(opencv_videoio.VideoCapture capture) {
        this.capture = capture;
        pMOG2 = opencv_video.createBackgroundSubtractorMOG2();
        pMOG2.setHistory(100);
        pknn = opencv_video.createBackgroundSubtractorKNN();
        pknn.setHistory(100);
        kde = new KernelDensityEstimator();
        kde.setN(10);
        vibe = new Vibe();
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

    public Mat getGaussianMixtureModel() {
        Mat frame = new Mat();
        Mat blurFrame = new Mat();
        if (!currentFrame.empty()) {
            opencv_imgproc.GaussianBlur(currentFrame, blurFrame, gaussianFilterSize, 0);
            pMOG2.apply(blurFrame, frame, 0.1);
        }
        return frame;
    }

    public Mat getKNNModel() {
        Mat frame = new Mat();
        Mat blurFrame = new Mat();
        if (!currentFrame.empty()) {
            opencv_imgproc.GaussianBlur(currentFrame, blurFrame, gaussianFilterSize, 0);
            pknn.apply(blurFrame, frame, 0.1);
        }
        return frame;
    }

    /*
    public Mat getKDEModel() {
        Mat frame = new Mat();
        Mat blurFrame = new Mat();
        if (!currentFrame.empty()) {
            opencv_imgproc.resize(currentFrame, blurFrame, new Size(currentFrame.arrayWidth() / 2, currentFrame.arrayHeight() / 2));
            opencv_imgproc.GaussianBlur(blurFrame, blurFrame, gaussianFilterSize, 0);

            try {
                frame = kde.foregroundMask(blurFrame);
            } catch (KernelDensityEstimator.KernelDensityEstimatorException e) {
                log("Error");
                e.printStackTrace();
            }
        }
        return frame;
    }

    public Mat getVibeModel() {
        Mat frame = new Mat();
        Mat blurFrame = new Mat();
        if (!currentFrame.empty()) {
            Imgproc.resize(currentFrame, blurFrame, new Size(currentFrame.width() / 2, currentFrame.height() / 2));
            Imgproc.GaussianBlur(blurFrame, blurFrame, gaussianFilterSize, 0);
            frame = vibe.foregroundMask(blurFrame);
        }
        return frame;
    }*/


    public Mat getGaussianBlur() {
        Mat blurFrame = new Mat();
        if (!currentFrame.empty()) {
            opencv_imgproc.GaussianBlur(currentFrame, blurFrame, gaussianFilterSize, 0);
        }
        return blurFrame;
    }

   /* public Mat getCanny() {
        //mat gray image holder
        Mat imageGray = new Mat();
        //mat canny image
        Mat imageCny = new Mat();
        if (!currentFrame.empty()) {
            //Convert the image in to gray image single channel image
            Imgproc.cvtColor(currentFrame, imageGray, Imgproc.COLOR_BGR2GRAY);
            //Canny Edge Detection
            Imgproc.Canny(imageGray, imageCny, 10, 100, 3, true);
        }
        return imageCny;
    }*/

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

    public void setHistoryGMM(int historyGMM) {
        log("Change GMM History to:" + historyGMM);
        pMOG2.setHistory(historyGMM);
    }

    public void setHistoryKNN(int historyKNN) {
        log("Change KNN History to:" + historyKNN);
        pknn.setHistory(historyKNN);
    }


    public void setGaussianFilterSize(int size) {
        int validSize = (size % 2) != 0 ? size : size - 1;
        log("Change Gaussian filter size to:" + validSize);
        gaussianFilterSize = new Size(validSize, validSize);
    }

    public void setKDEThreshole(double threshole) {
        log("Change KDE Threshold to:" + threshole);
        kde.setThreshold(threshole);
    }

    public void setVibeThreshole(double threshole) {
        log("Change Vibe Threshold to:" + threshole);
        vibe.model.setMatchingThreshold(threshole);
    }


    private void log(Object o) {
        System.out.println(o);
    }
}
