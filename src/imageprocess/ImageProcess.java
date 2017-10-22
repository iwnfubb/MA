package imageprocess;

import algorithms.KernelDensityEstimator;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractor;
import org.opencv.video.BackgroundSubtractorKNN;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;

public class ImageProcess {
    private VideoCapture capture;
    private Mat staticBackground = new Mat();
    private Mat currentFrame = new Mat();
    private BackgroundSubtractorMOG2 pMOG2;
    private BackgroundSubtractorKNN pknn;
    private Size gaussianFilterSize = (new Size(3, 3));
    KernelDensityEstimator kde;

    public ImageProcess(VideoCapture capture) {
        this.capture = capture;
        pMOG2 = Video.createBackgroundSubtractorMOG2();
        pMOG2.setHistory(100);
        pknn = Video.createBackgroundSubtractorKNN();
        pknn.setHistory(100);
        kde = new KernelDensityEstimator();
        kde.setN(5);
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
            Imgproc.GaussianBlur(currentFrame, blurFrame, gaussianFilterSize, 0);
            pMOG2.apply(blurFrame, frame, 0.1);
        }
        return frame;
    }

    public Mat getKNNModel() {
        Mat frame = new Mat();
        Mat blurFrame = new Mat();
        if (!currentFrame.empty()) {
            Imgproc.GaussianBlur(currentFrame, blurFrame, gaussianFilterSize, 0);
            pknn.apply(blurFrame, frame, 0.1);
        }
        return frame;
    }

    public Mat getKDEModel() {
        Mat frame = new Mat();
        Mat blurFrame = new Mat();
        if (!currentFrame.empty()) {
            Imgproc.resize(currentFrame, blurFrame, new Size(currentFrame.width() / 2, currentFrame.height() / 2));
            Imgproc.GaussianBlur(blurFrame, blurFrame, gaussianFilterSize, 0);

            try {
                frame = kde.foregroundMask(blurFrame);
            } catch (KernelDensityEstimator.KernelDensityEstimatorException e) {
                log("Error");
                e.printStackTrace();
            }
        }
        return frame;
    }


    public Mat getGaussianBlur() {
        Mat blurFrame = new Mat();
        if (!currentFrame.empty()) {
            Imgproc.GaussianBlur(currentFrame, blurFrame, gaussianFilterSize, 0);
        }
        return blurFrame;
    }

    public Mat getCanny() {
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


    private void log(Object o) {
        System.out.println(o);
    }
}
