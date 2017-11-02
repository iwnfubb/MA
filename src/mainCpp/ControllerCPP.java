package mainCpp;

import com.sun.javafx.geom.Vec3d;
import imageprocess.Utils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Slider;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.indexer.UByteBufferIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_videoio.VideoCapture;
import org.bytedeco.javacpp.opencv_xfeatures2d;
import org.bytedeco.javacpp.opencv_features2d;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_video;
import org.bytedeco.javacpp.opencv_ml;
import org.opencv.core.Point;

import javax.naming.SizeLimitExceededException;
import java.nio.Buffer;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class ControllerCPP {
    @FXML
    private ImageView currentFrameView;
    @FXML
    private ImageView surfImgView;
    @FXML
    private ImageView gaussianBlurView;
    @FXML
    private ImageView opticalFlowView;
    @FXML
    private ImageView gmmWeightView;
    @FXML
    private ImageView gmmMeansView;
    @FXML
    private CheckBox opticalFlowActive;
    @FXML
    private CheckBox surfImgActive;
    @FXML
    private CheckBox gmmClusterActive;
    @FXML
    private Slider gaussianBlur;
    @FXML
    private Slider hessianThreshold;
    @FXML
    private Slider nOctaveLayer;
    @FXML
    private Button button;

    // a timer for acquiring the video stream
    private ScheduledExecutorService timer;
    // the OpenCV object that realizes the video capture
    private VideoCapture capture = new VideoCapture();
    ImageProcessCPP imgProcess = new ImageProcessCPP(this.capture);
    // a flag to change the button behavior
    private boolean cameraActive = false;
    // the id of the camera to be used
    private static int cameraId = 1;

    private Mat prevgray = new Mat();

    /**
     * The action triggered by pushing the button on the GUI
     *
     * @param event the push button event
     */
    @FXML
    protected void startCamera(ActionEvent event) {
        ini();
        if (!this.cameraActive) {
            // start the video capture
            this.capture.open(cameraId);

            // is the video stream available?
            if (this.capture.isOpened()) {
                this.cameraActive = true;

                // grab a frame every 33 ms (30 frames/sec)
                Runnable frameGrabber = () -> {
                    // effectively grab and process a single frame
                    Mat originalFrame = imgProcess.getOriginalFrame();
                    Mat gaussianBlurFrame = imgProcess.getGaussianBlur();
                    Image imageToShow = Utils.mat2Image(originalFrame);
                    updateImageView(currentFrameView, imageToShow);

                    if (!gaussianBlurFrame.empty()) {
                        Image mmgImageToShow = Utils.mat2Image(gaussianBlurFrame);
                        updateImageView(gaussianBlurView, mmgImageToShow);
                    }

                    if (surfImgActive.isSelected() && !gaussianBlurFrame.empty()) {
                        Mat surfImg = new Mat();
                        opencv_core.KeyPointVector surfKeyPoint = imgProcess.getSURFKeyPoint(gaussianBlurFrame, new Mat());
                        opencv_features2d.drawKeypoints(gaussianBlurFrame, surfKeyPoint, surfImg, new opencv_core.Scalar(0, 0, 255, 0), 4);
                        Image mmgImageToShow = Utils.mat2Image(surfImg);
                        updateImageView(surfImgView, mmgImageToShow);
                    }

                    if (opticalFlowActive.isSelected() && !gaussianBlurFrame.empty()) {
                        Mat flow = new Mat(), img = new Mat();
                        Mat flowUmat = new Mat();
                        Indexer indexer1 = originalFrame.createIndexer();
                        originalFrame.copyTo(img);
                        opencv_imgproc.cvtColor(img, img, opencv_imgproc.COLOR_BGR2GRAY);

                        if (!prevgray.empty()) {
                            opencv_video.calcOpticalFlowFarneback(prevgray, img, flowUmat, 0.4, 1, 12, 2, 8, 1.5, 0);
                            flowUmat.copyTo(flow);
                            FloatRawIndexer indexer = flow.createIndexer();
                            for (int y = 0; y < originalFrame.rows(); y += 20)
                                for (int x = 0; x < originalFrame.cols(); x += 20) {
                                    //flow.get(x, y)
                                    float flowatx = indexer.get(y, x, 0) * 10;
                                    float flowaty = indexer.get(y, x, 1) * 10;
                                    opencv_imgproc.line(originalFrame,
                                            new opencv_core.Point(x, y),
                                            new opencv_core.Point(Math.round(x + flowatx), Math.round(y + flowaty)),
                                            new opencv_core.Scalar(0, 255, 0, 0));
                                    opencv_imgproc.circle(originalFrame,
                                            new opencv_core.Point(x, y),
                                            2,
                                            new opencv_core.Scalar(0, 0, 0, 0), -2, 4, 0);
                                }
                            img.copyTo(prevgray);
                        } else {
                            img.copyTo(prevgray);

                        }

                        Image mmgImageToShow = Utils.mat2Image(originalFrame);
                        updateImageView(opticalFlowView, mmgImageToShow);
                    }

                    if (!gaussianBlurFrame.empty()) {
                        opencv_ml.EM em = opencv_ml.EM.create();
                        Mat labels = new Mat();
                        Mat probs = new Mat();
                        Mat samples = new Mat();
                        gaussianBlurFrame.reshape(1, gaussianBlurFrame.rows() * gaussianBlurFrame.cols()).convertTo(samples, opencv_core.CV_32FC1, 1.0 / 255.0, 0.0);
                        //em.train(samples, 3, labels);
                        em.trainEM(samples, new Mat(), labels, probs);
                        probs.reshape(5, gaussianBlurFrame.rows()).convertTo(probs, opencv_core.CV_8UC3, 255.0, 0);
                        Image mmgImageToShow = Utils.mat2Image(probs);
                        updateImageView(gmmWeightView, mmgImageToShow);
                    }
                };

                this.timer = Executors.newSingleThreadScheduledExecutor();
                this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

                // update the button content
                this.button.setText("Stop Camera");
            } else {
                // log the error
                System.err.println("Impossible to open the camera connection...");
            }
        } else {
            // the camera is not active at this point
            this.cameraActive = false;
            // update again the button content
            this.button.setText("Start Camera");

            // stop the timer
            this.stopAcquisition();
        }
    }

    @FXML
    protected void setStaticBackground(ActionEvent event) {
        imgProcess.setStaticBackground();
    }

    private void stopAcquisition() {
        if (this.timer != null && !this.timer.isShutdown()) {
            try {
                // stop the timer
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                // log any exception
                System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
            }
        }

        if (this.capture.isOpened()) {
            // release the camera
            this.capture.release();
        }
    }

    private void updateImageView(ImageView view, Image image) {
        Utils.onFXThread(view.imageProperty(), image);
    }

    protected void setClosed() {
        this.stopAcquisition();
    }

    private void ini() {
        gaussianBlur.setMin(1);
        gaussianBlur.setMax(45);
        gaussianBlur.valueProperty().addListener((observable, oldValue, newValue) -> {
            imgProcess.setGaussianFilterSize(newValue.intValue());
        });

        hessianThreshold.setMin(1);
        hessianThreshold.setMax(10000);
        hessianThreshold.valueProperty().addListener((observable, oldValue, newValue) -> {
            imgProcess.setHessianThreshold(newValue.intValue());
        });

        nOctaveLayer.setMin(1);
        nOctaveLayer.setMax(45);
        nOctaveLayer.valueProperty().addListener((observable, oldValue, newValue) -> {
            imgProcess.setNOctaveLayer(newValue.intValue());
        });
    }

}
