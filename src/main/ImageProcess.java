package main;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;

public class ImageProcess {
    private VideoCapture capture;
    private Mat staticBackground = new Mat();

    public ImageProcess(VideoCapture capture) {
        this.capture = capture;
    }

    public Mat getOriginalFrame() {
        Mat frame = new Mat();
        if (this.capture.isOpened()) {
            try {
                this.capture.read(frame);
                if (!frame.empty()) {
                    //Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
                }
            } catch (Exception e) {
                System.err.println("Exception during the image elaboration: " + e);
            }
        }
        return frame;
    }

    public Mat getGaussianMixtureModel() {
        Mat frame = new Mat();
        if (this.capture.isOpened()) {
            try {
                this.capture.read(frame);
                if (!frame.empty()) {
                    BackgroundSubtractorMOG2 pMOG2 = Video.createBackgroundSubtractorMOG2();
                    if (!staticBackground.empty()) {
                        setStaticBackground();
                        pMOG2.apply(staticBackground, frame);
                    }
                }
            } catch (Exception e) {
                System.err.println("Exception during the image elaboration: " + e);
            }
        }
        return frame;
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
}
