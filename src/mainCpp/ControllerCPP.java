package mainCpp;

import imageprocess.Utils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_features2d;
import org.bytedeco.javacpp.opencv_videoio.VideoCapture;

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
    private ImageView grabcutView;
    @FXML
    private CheckBox opticalFlowActive;
    @FXML
    private CheckBox surfImgActive;
    @FXML
    private CheckBox clusteringActive;
    @FXML
    private CheckBox grabcutActive;
    @FXML
    private Slider gaussianBlur;
    @FXML
    private Slider hessianThreshold;
    @FXML
    private Slider nOctaveLayer;
    @FXML
    private Button button;
    @FXML
    private TextField epsilon;
    @FXML
    private TextField minPoints;

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

                    opencv_core.KeyPointVector surfKeyPoint = new opencv_core.KeyPointVector();
                    if (surfImgActive.isSelected() && !gaussianBlurFrame.empty()) {
                        Mat surfImg = new Mat();
                        surfKeyPoint = imgProcess.getSURFKeyPoint(gaussianBlurFrame, new Mat());
                        opencv_features2d.drawKeypoints(gaussianBlurFrame, surfKeyPoint, surfImg, new opencv_core.Scalar(0, 0, 255, 0), 4);
                        Image mmgImageToShow = Utils.mat2Image(surfImg);
                        updateImageView(surfImgView, mmgImageToShow);
                    }
                    Mat flow = new Mat();
                    if (opticalFlowActive.isSelected() && !gaussianBlurFrame.empty()) {
                        Mat ofFrame = imgProcess.opticalFLow(gaussianBlurFrame, flow);
                        Image mmgImageToShow = Utils.mat2Image(ofFrame);
                        updateImageView(opticalFlowView, mmgImageToShow);
                    }
                    if (!gaussianBlurFrame.empty() && clusteringActive.isSelected() && surfKeyPoint.size() != 0) {
                        Mat clusteringCoordinateGMM = imgProcess.clusteringCoordinateGMM(gaussianBlurFrame, surfKeyPoint);
                        Image mmgImageToShow = Utils.mat2Image(clusteringCoordinateGMM);
                        updateImageView(gmmWeightView, mmgImageToShow);
                    }
                    if (!gaussianBlurFrame.empty() && clusteringActive.isSelected() && surfKeyPoint.size() != 0) {
                        //clusteringCoordinateKmeans(surfKeyPoint, originalFrame, flow);
                        double eps;
                        int minP;
                        try {
                            eps = Double.parseDouble(epsilon.getText());
                            minP = Integer.parseInt(minPoints.getText());
                        } catch (NumberFormatException e) {
                            System.out.println("Error by Parsing String");
                            eps = 0.05d;
                            minP = 10;
                        }
                        Mat classificationFrame = imgProcess.clusteringCoordinateDBSCAN(gaussianBlurFrame, surfKeyPoint, eps, minP);
                        Image mmgImageToShow = Utils.mat2Image(classificationFrame);
                        updateImageView(gmmMeansView, mmgImageToShow);
                    }
                    if (!gaussianBlurFrame.empty() && surfKeyPoint.size() != 0 && grabcutActive.isSelected() && !flow.empty()) {
                        Mat grabcutFrame = imgProcess.proposedModel(gaussianBlurFrame, surfKeyPoint, flow);
                        Image mmgImageToShow = Utils.mat2Image(grabcutFrame);
                        updateImageView(grabcutView, mmgImageToShow);

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
