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
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.indexer.UByteBufferIndexer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_videoio.VideoCapture;
import org.opencv.core.Point;

import javax.naming.SizeLimitExceededException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
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
    private CheckBox clusteringActive;

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

                    opencv_core.KeyPointVector surfKeyPoint = new opencv_core.KeyPointVector();
                    if (surfImgActive.isSelected() && !gaussianBlurFrame.empty()) {
                        Mat surfImg = new Mat();
                        surfKeyPoint = imgProcess.getSURFKeyPoint(gaussianBlurFrame, new Mat());
                        opencv_features2d.drawKeypoints(gaussianBlurFrame, surfKeyPoint, surfImg, new opencv_core.Scalar(0, 0, 255, 0), 4);
                        Image mmgImageToShow = Utils.mat2Image(surfImg);
                        updateImageView(surfImgView, mmgImageToShow);
                    }

                    if (opticalFlowActive.isSelected() && !gaussianBlurFrame.empty()) {
                        Mat flow = new Mat(), img = new Mat(), copyOfOriginal = new Mat();
                        Mat flowUmat = new Mat();
                        originalFrame.copyTo(img);
                        originalFrame.copyTo(copyOfOriginal);
                        opencv_imgproc.cvtColor(img, img, opencv_imgproc.COLOR_BGR2GRAY);

                        if (!prevgray.empty()) {
                            opencv_video.calcOpticalFlowFarneback(prevgray, img, flowUmat, 0.4, 1, 12, 2, 8, 1.5, 0);
                            flowUmat.copyTo(flow);
                            FloatRawIndexer indexer = flow.createIndexer();
                            for (int y = 0; y < copyOfOriginal.rows(); y += 20)
                                for (int x = 0; x < copyOfOriginal.cols(); x += 20) {
                                    //flow.get(x, y)
                                    float flowatx = indexer.get(y, x, 0) * 10;
                                    float flowaty = indexer.get(y, x, 1) * 10;
                                    opencv_imgproc.line(copyOfOriginal,
                                            new opencv_core.Point(x, y),
                                            new opencv_core.Point(Math.round(x + flowatx), Math.round(y + flowaty)),
                                            new opencv_core.Scalar(0, 255, 0, 0));
                                    opencv_imgproc.circle(copyOfOriginal,
                                            new opencv_core.Point(x, y),
                                            2,
                                            new opencv_core.Scalar(0, 0, 0, 0), -2, 4, 0);
                                }
                            img.copyTo(prevgray);
                        } else {
                            img.copyTo(prevgray);

                        }

                        Image mmgImageToShow = Utils.mat2Image(copyOfOriginal);
                        updateImageView(opticalFlowView, mmgImageToShow);
                    }

                    if (!gaussianBlurFrame.empty() && clusteringActive.isSelected() && surfKeyPoint.size() != 0) {
                        System.out.println("Starting Clustering Texture...");
                        long startTime = System.currentTimeMillis();
                        Mat labels = new Mat();
                        Mat probs = new Mat();
                        Mat copyOfOriginal = new Mat();
                        originalFrame.copyTo(copyOfOriginal);
                        Mat samples = new Mat(new opencv_core.Size(3, (int) surfKeyPoint.size()), opencv_core.CV_8UC1);
                        for (int i = 0; i < surfKeyPoint.size(); i++) {
                            opencv_core.KeyPoint keyPoint = surfKeyPoint.get(i);
                            byte b = (copyOfOriginal.ptr((int) keyPoint.pt().y(), (int) keyPoint.pt().x()).get(0));
                            byte g = (copyOfOriginal.ptr((int) keyPoint.pt().y(), (int) keyPoint.pt().x()).get(1));
                            byte r = (copyOfOriginal.ptr((int) keyPoint.pt().y(), (int) keyPoint.pt().x()).get(2));
                            samples.ptr(i, 0).put(0, b);
                            samples.ptr(i, 1).put(0, g);
                            samples.ptr(i, 2).put(0, r);
                        }
                        System.out.println("Done1");


                        opencv_ml.EM em = opencv_ml.EM.create();
                        em.setClustersNumber(3);
                        //gaussianBlurFrame.reshape(1, gaussianBlurFrame.rows() * gaussianBlurFrame.cols()).convertTo(samples, opencv_core.CV_32FC1, 1.0 / 255.0, 0.0);
                        samples.convertTo(samples, opencv_core.CV_32FC1, 1.0 / 255.0, 0.0);
                        //em.train(samples, 3, labels);
                        em.trainEM(samples, new Mat(), labels, probs);
                        System.out.println("Done2");

                        for (int i = 0; i < surfKeyPoint.size(); i++) {
                            opencv_core.KeyPoint keyPoint = surfKeyPoint.get(i);
                            opencv_core.Scalar scalar;
                            if (labels.ptr(i, 0).get(0) == 0)
                                scalar = new opencv_core.Scalar(255, 0, 0, 0);
                            else if (labels.ptr(i, 0).get(0) == 1)
                                scalar = new opencv_core.Scalar(0, 255, 0, 0);
                            else
                                scalar = new opencv_core.Scalar(0, 0, 255, 0);
                            opencv_imgproc.circle(copyOfOriginal,
                                    new opencv_core.Point((int) keyPoint.pt().x(), (int) keyPoint.pt().y()),
                                    5,
                                    scalar, -5, 4, 0);
                        }
                        System.out.println("Done3");
                        Image mmgImageToShow = Utils.mat2Image(copyOfOriginal);
                        updateImageView(gmmWeightView, mmgImageToShow);
                        System.out.println("Clustering Time:" + (System.currentTimeMillis() - startTime));
                    }

                    if (!gaussianBlurFrame.empty() && clusteringActive.isSelected() && surfKeyPoint.size() != 0) {
                        System.out.println("Starting Clustering Position ...");
                        long startTime = System.currentTimeMillis();
                        Mat labels = new Mat();
                        Mat probs = new Mat();
                        Mat copyOfOriginal = new Mat();
                        originalFrame.copyTo(copyOfOriginal);
                        float width = copyOfOriginal.arrayWidth();
                        float height = copyOfOriginal.arrayHeight();
                        Mat samples = new Mat(new opencv_core.Size(2, (int) surfKeyPoint.size()), opencv_core.CV_32FC1);
                        for (int i = 0; i < surfKeyPoint.size(); i++) {
                            opencv_core.KeyPoint keyPoint = surfKeyPoint.get(i);
                            float x = keyPoint.pt().x();
                            float y = keyPoint.pt().y();
                            samples.ptr(i, 0).put(float2ByteArray(x / width));
                            samples.ptr(i, 1).put(float2ByteArray(y / height));
                        }
                        System.out.println("Done1");

                        opencv_ml.EM em = opencv_ml.EM.create();
                        em.setClustersNumber(3);
                        //gaussianBlurFrame.reshape(1, gaussianBlurFrame.rows() * gaussianBlurFrame.cols()).convertTo(samples, opencv_core.CV_32FC1, 1.0 / 255.0, 0.0);
                        //samples.convertTo(samples, opencv_core.CV_32FC1, 1.0, 0.0);
                        //em.train(samples, 3, labels);
                        em.trainEM(samples, new Mat(), labels, probs);
                        System.out.println("Done2");

                        for (int i = 0; i < surfKeyPoint.size(); i++) {
                            opencv_core.KeyPoint keyPoint = surfKeyPoint.get(i);
                            opencv_core.Scalar scalar;
                            if (labels.ptr(i, 0).get(0) == 0)
                                scalar = new opencv_core.Scalar(255, 0, 0, 0);
                            else if (labels.ptr(i, 0).get(0) == 1)
                                scalar = new opencv_core.Scalar(0, 255, 0, 0);
                            else
                                scalar = new opencv_core.Scalar(0, 0, 255, 0);
                            opencv_imgproc.circle(copyOfOriginal,
                                    new opencv_core.Point((int) keyPoint.pt().x(), (int) keyPoint.pt().y()),
                                    5,
                                    scalar, -5, 4, 0);

                        }
                        System.out.println("Done3");

                        Image mmgImageToShow = Utils.mat2Image(copyOfOriginal);
                        updateImageView(gmmMeansView, mmgImageToShow);

                        System.out.println("Clustering Time:" + (System.currentTimeMillis() - startTime));
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

    private static byte[] float2ByteArray(float value) {
        byte[] array = ByteBuffer.allocate(4).putFloat(value).order(ByteOrder.LITTLE_ENDIAN).array();
        byte[] result = new byte[4];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[array.length - i - 1];
        }
        return result;
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
