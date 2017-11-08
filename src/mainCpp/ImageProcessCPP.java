package mainCpp;

import net.sf.javaml.clustering.OPTICS;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.core.SparseInstance;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Size;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;

public class ImageProcessCPP {
    private opencv_videoio.VideoCapture capture;
    private Mat staticBackground = new Mat();
    private Mat currentFrame = new Mat();
    private Size gaussianFilterSize = (new Size(3, 3));
    opencv_xfeatures2d.SURF surf;
    private Mat prevgray = new Mat();

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

    public Mat opticalFLow(Mat input, Mat flow) {
        Mat img = new Mat(), copyOfOriginal = new Mat();
        Mat flowUmat = new Mat();
        input.copyTo(img);
        input.copyTo(copyOfOriginal);
        opencv_imgproc.cvtColor(img, img, opencv_imgproc.COLOR_BGR2GRAY);

        if (!prevgray.empty()) {
            opencv_video.calcOpticalFlowFarneback(prevgray, img, flowUmat, 0.4, 1, 12, 2, 8, 1.5, 0);
            flowUmat.copyTo(flow);
            FloatRawIndexer indexer = flow.createIndexer();
            for (int y = 0; y < copyOfOriginal.rows(); y += 10)
                for (int x = 0; x < copyOfOriginal.cols(); x += 10) {
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
        return copyOfOriginal;
    }

    public Mat clusteringCoordinateDBSCAN(opencv_core.Mat input, opencv_core.KeyPointVector surfKeyPoint, double eps, int minP) {
        System.out.println("Starting Clustering Position DBSCAN ...");
        long startTime = System.currentTimeMillis();
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);
        Dataset data = new DefaultDataset();
        for (int i = 0; i < surfKeyPoint.size(); i++) {
            opencv_core.KeyPoint keyPoint = surfKeyPoint.get(i);
            float x = keyPoint.pt().x();
            float y = keyPoint.pt().y();
            //Create instance with 2 Attributes
            Instance instance = new SparseInstance(2);
            instance.put(1, Double.parseDouble(Float.toString(x)));
            instance.put(2, Double.parseDouble(Float.toString(y)));
            data.add(instance);
        }
        System.out.println("Done1");

        //===== OPTIC =====
        OPTICS optics = new OPTICS(eps, minP);
        Dataset[] cluster = optics.cluster(data);

        //===== DBSCAN =====
        //DensityBasedSpatialClustering dbscan = new DensityBasedSpatialClustering(eps, minP);
        //Dataset[] cluster = dbscan.cluster(data);s

        System.out.println("Done2");
        for (int i = 0; i < cluster.length; i++) {
            for (int index = 0; index < cluster[i].size(); index++) {
                Instance instance = cluster[i].get(index);
                opencv_core.Scalar scalar;
                if (i == 0) {
                    scalar = new opencv_core.Scalar(0, 255, 0, 0);
                } else if (i == 1) {
                    scalar = new opencv_core.Scalar(0, 0, 255, 0);
                } else {
                    scalar = new opencv_core.Scalar(255, 0, 0, 0);
                }
                opencv_imgproc.circle(copyOfOriginal,
                        new opencv_core.Point((int) instance.value(1), (int) instance.value(2)),
                        5,
                        scalar, -5, 4, 0);

            }
        }
        System.out.println("Done3");
        System.out.println("Clustering Time:" + (System.currentTimeMillis() - startTime));
        return copyOfOriginal;
    }

    private Mat clusteringTexture(Mat input, opencv_core.KeyPointVector surfKeyPoint) {
        System.out.println("Starting Clustering Texture...");
        long startTime = System.currentTimeMillis();
        Mat labels = new Mat();
        Mat probs = new Mat();
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);
        Mat samples = new Mat(new Size(3, (int) surfKeyPoint.size()), opencv_core.CV_8UC1);
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
            if (labels.ptr(i, 0).get(0) == 0) {
                scalar = new opencv_core.Scalar(255, 0, 0, 0);
            } else if (labels.ptr(i, 0).get(0) == 1) {
                scalar = new opencv_core.Scalar(0, 255, 0, 0);
            } else {
                scalar = new opencv_core.Scalar(0, 0, 255, 0);
            }
            opencv_imgproc.circle(copyOfOriginal,
                    new opencv_core.Point((int) keyPoint.pt().x(), (int) keyPoint.pt().y()),
                    5,
                    scalar, -5, 4, 0);
        }
        System.out.println("Done3");
        System.out.println("Clustering Time:" + (System.currentTimeMillis() - startTime));
        return copyOfOriginal;
    }

    public Mat clusteringCoordinateKmeans(opencv_core.KeyPointVector surfKeyPoint, opencv_core.Mat input,
                                          opencv_core.Mat flow) {
        System.out.println("Starting Clustering Position Kmeans ...");
        long startTime = System.currentTimeMillis();
        Mat labels = new Mat(new opencv_core.Size(1, (int) surfKeyPoint.size()), opencv_core.CV_32S);
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);
        float width = copyOfOriginal.arrayWidth();
        float height = copyOfOriginal.arrayHeight();
        Mat samples = new Mat(new opencv_core.Size(2, (int) surfKeyPoint.size()), opencv_core.CV_32F);
        FloatRawIndexer indexer = flow.createIndexer();
        for (int i = 0; i < surfKeyPoint.size(); i++) {
            opencv_core.KeyPoint keyPoint = surfKeyPoint.get(i);
            float x = keyPoint.pt().x();
            float y = keyPoint.pt().y();
            samples.ptr(i, 0).put(float2ByteArray(x));
            samples.ptr(i, 1).put(float2ByteArray(y));
            if (Math.abs(indexer.get((int) y, (int) x, 0)) > 1.0
                    || Math.abs(indexer.get((int) y, (int) x, 1)) > 1.0) {
                labels.ptr(i, 0).put(new byte[]{0, 0, 0, 0});
            } else {
                labels.ptr(i, 0).put(new byte[]{1, 0, 0, 0});
            }
        }
        System.out.println("Done1");
        Mat centers = new Mat();
        opencv_core.TermCriteria criteria = new opencv_core.TermCriteria(
                opencv_core.TermCriteria.COUNT, 100, 1);
        opencv_core.kmeans(samples, 2, labels, criteria, 3, opencv_core.KMEANS_USE_INITIAL_LABELS, centers);
        System.out.println("Done2");

        for (int i = 0; i < surfKeyPoint.size(); i++) {
            opencv_core.KeyPoint keyPoint = surfKeyPoint.get(i);
            opencv_core.Scalar scalar;
            if (labels.ptr(i, 0).get(0) == 0) {
                scalar = new opencv_core.Scalar(255, 0, 0, 0);
            } else if (labels.ptr(i, 0).get(0) == 1) {
                scalar = new opencv_core.Scalar(0, 255, 0, 0);
            } else {
                scalar = new opencv_core.Scalar(0, 0, 255, 0);
            }
            opencv_imgproc.circle(copyOfOriginal,
                    new opencv_core.Point((int) keyPoint.pt().x(), (int) keyPoint.pt().y()),
                    5,
                    scalar, -5, 4, 0);

        }
        System.out.println("Done3");
        System.out.println("Clustering Time:" + (System.currentTimeMillis() - startTime));
        return copyOfOriginal;
    }

    public Mat clusteringCoordinateGMM(opencv_core.Mat input, opencv_core.KeyPointVector surfKeyPoint) {
        System.out.println("Starting Clustering Position GMM...");
        long startTime = System.currentTimeMillis();
        Mat labels = new Mat();
        Mat probs = new Mat();
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);
        float width = copyOfOriginal.arrayWidth();
        float height = copyOfOriginal.arrayHeight();
        Mat samples = new Mat(new opencv_core.Size(2, (int) surfKeyPoint.size()), opencv_core.CV_32FC1);
        for (int i = 0; i < surfKeyPoint.size(); i++) {
            opencv_core.KeyPoint keyPoint = surfKeyPoint.get(i);
            float x = keyPoint.pt().x();
            float y = keyPoint.pt().y();
            samples.ptr(i, 0).put(ImageProcessCPP.float2ByteArray(x / width));
            samples.ptr(i, 1).put(ImageProcessCPP.float2ByteArray(y / height));
        }
        System.out.println("Done1");
        opencv_ml.EM em = opencv_ml.EM.create();
        em.setClustersNumber(3);
        em.setTermCriteria(new opencv_core.TermCriteria(
                opencv_core.TermCriteria.COUNT, 100, 1));
        em.trainEM(samples, new Mat(), labels, probs);
        System.out.println("Done2");

        for (int i = 0; i < surfKeyPoint.size(); i++) {
            opencv_core.KeyPoint keyPoint = surfKeyPoint.get(i);
            opencv_core.Scalar scalar;
            if (labels.ptr(i, 0).get(0) == 0) {
                scalar = new opencv_core.Scalar(255, 0, 0, 0);
            } else if (labels.ptr(i, 0).get(0) == 1) {
                scalar = new opencv_core.Scalar(0, 255, 0, 0);
            } else {
                scalar = new opencv_core.Scalar(0, 0, 255, 0);
            }
            opencv_imgproc.circle(copyOfOriginal,
                    new opencv_core.Point((int) keyPoint.pt().x(), (int) keyPoint.pt().y()),
                    5,
                    scalar, -5, 4, 0);

        }
        System.out.println("Done3");
        System.out.println("Clustering Time:" + (System.currentTimeMillis() - startTime));
        return copyOfOriginal;
    }

    public Mat grabCut(Mat input, opencv_core.KeyPointVector surf, Mat flow) {
        long start = System.currentTimeMillis();
        log("Start Grabcut ... ");
        Mat mask = new Mat(input.size(), opencv_core.CV_8UC1, opencv_core.Scalar.all(opencv_imgproc.GC_PR_BGD));
        Mat bgModel = new Mat(new Size(65, 1), opencv_core.CV_64FC1, opencv_core.Scalar.all(0));
        Mat fgModel = new Mat(new Size(65, 1), opencv_core.CV_64FC1, opencv_core.Scalar.all(0));
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);
        for (int i = 0; i < surf.size(); i++) {
            opencv_core.Point2f pt = surf.get(i).pt();
            mask.ptr((int) pt.y(), (int) pt.x()).put(0, (byte) opencv_imgproc.GC_PR_FGD);
        }

        opencv_imgproc.grabCut(input, mask,
                new opencv_core.Rect(20, 20, input.cols() - 20, input.rows() - 20),
                bgModel, fgModel, 5,
                opencv_imgproc.GC_INIT_WITH_MASK);
        log("Stop Grabcut ... ");
        log("Time :" + (System.currentTimeMillis() - start));
        return mergeImageAndMask(copyOfOriginal, mask);
    }

    Mat backGroundModel;
    boolean initBackgroundModel = false;

    public Mat proposedModel(Mat input, opencv_core.KeyPointVector surf, Mat flow) {
        if (!initBackgroundModel) {
            backGroundModel = new Mat(input.size(), opencv_core.CV_8UC1, opencv_core.Scalar.all(opencv_imgproc.GC_PR_BGD));
            initBackgroundModel = true;
            return input;
        }
        long start = System.currentTimeMillis();
        //update background model
        log("Update background model ... ");
        FloatRawIndexer indexer = flow.createIndexer();
        for (int i = 0; i < surf.size(); i++) {
            opencv_core.Point2f pt = surf.get(i).pt();
            float floatAtX = indexer.get((int) pt.y(), (int) pt.x(), 0);
            float floatAtY = indexer.get((int) pt.y(), (int) pt.x(), 0);
            BytePointer ptr = backGroundModel.ptr((int) pt.y(), (int) pt.x());
            if (Math.abs(floatAtX) > 1.0f && Math.abs(floatAtY) > 1.0f) {
                if (ptr.get(0) == (byte) opencv_imgproc.GC_PR_FGD)
                    ptr.put(0, (byte) opencv_imgproc.GC_FGD);
                else if (ptr.get(0) == (byte) opencv_imgproc.GC_PR_BGD)
                    ptr.put(0, (byte) opencv_imgproc.GC_PR_FGD);
            } else {
                if (ptr.get(0) == (byte) opencv_imgproc.GC_FGD)
                    ptr.put(0, (byte) opencv_imgproc.GC_PR_FGD);
                else if (ptr.get(0) == (byte) opencv_imgproc.GC_PR_FGD)
                    ptr.put(0, (byte) opencv_imgproc.GC_PR_BGD);
            }
        }

        log("Start Grabcut ... ");
        Mat bgModel = new Mat(new Size(65, 1), opencv_core.CV_64FC1, opencv_core.Scalar.all(0));
        Mat fgModel = new Mat(new Size(65, 1), opencv_core.CV_64FC1, opencv_core.Scalar.all(0));
        Mat mask = new Mat();
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);
        backGroundModel.copyTo(mask);
        opencv_imgproc.grabCut(input, mask,
                new opencv_core.Rect(20, 20, input.cols() - 20, input.rows() - 20),
                bgModel, fgModel, 5,
                opencv_imgproc.GC_INIT_WITH_MASK);
        log("Stop Grabcut ... ");
        log("Time :" + (System.currentTimeMillis() - start));
        return mergeImageAndMask(copyOfOriginal, mask);
    }


    Mat previousFrame = new Mat();
    Mat previousFrameDescriptors = new Mat();
    opencv_core.KeyPointVector previousKeyPoint = new opencv_core.KeyPointVector();

    public Mat tobiModel(Mat input, opencv_core.KeyPointVector surfKeyPoints, Mat flow) {
        if (!initBackgroundModel) {
            backGroundModel = new Mat(input.size(), opencv_core.CV_8UC1, opencv_core.Scalar.all(opencv_imgproc.GC_PR_BGD));
            input.copyTo(previousFrame);
            surf.compute(previousFrame, surfKeyPoints, previousFrameDescriptors);
            previousKeyPoint.put(surfKeyPoints);
            initBackgroundModel = true;
            return input;
        }
        long start = System.currentTimeMillis();
        //update background model
        log("Update background model ... ");
        FloatRawIndexer indexer = flow.createIndexer();
        for (int i = 0; i < surfKeyPoints.size(); i++) {
            opencv_core.Point2f pt = surfKeyPoints.get(i).pt();
            float floatAtX = indexer.get((int) pt.y(), (int) pt.x(), 0);
            float floatAtY = indexer.get((int) pt.y(), (int) pt.x(), 0);
            BytePointer ptr = backGroundModel.ptr((int) pt.y(), (int) pt.x());
            if (Math.abs(floatAtX) > 1.0f && Math.abs(floatAtY) > 1.0f) {
                if (ptr.get(0) == (byte) opencv_imgproc.GC_PR_FGD)
                    ptr.put(0, (byte) opencv_imgproc.GC_FGD);
                else if (ptr.get(0) == (byte) opencv_imgproc.GC_PR_BGD)
                    ptr.put(0, (byte) opencv_imgproc.GC_PR_FGD);
            } else {
                if (ptr.get(0) == (byte) opencv_imgproc.GC_FGD)
                    ptr.put(0, (byte) opencv_imgproc.GC_PR_FGD);
                else if (ptr.get(0) == (byte) opencv_imgproc.GC_PR_FGD)
                    ptr.put(0, (byte) opencv_imgproc.GC_PR_BGD);
            }
        }

        log("Matching previous Frame to Current frame... ");
        Mat currentFrameDescriptors = new Mat();
        surf.compute(input, surfKeyPoints, currentFrameDescriptors);


        opencv_features2d.FlannBasedMatcher matcher = new opencv_features2d.FlannBasedMatcher();

        opencv_core.DMatchVector matches = new opencv_core.DMatchVector();
        matcher.match(currentFrameDescriptors, previousFrameDescriptors, matches);

        double max_dist = 0;
        double min_dist = 100;
        //-- Quick calculation of max and min distances between keypoints
        for (int i = 0; i < currentFrameDescriptors.rows(); i++) {
            double dist = matches.get(i).distance();
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }

        ArrayList<opencv_core.DMatch> goodMatches = new ArrayList<>();
        for (int i = 0; i < currentFrameDescriptors.rows(); i++) {
            if (matches.get(i).distance() <= Math.max(2 * min_dist, 0.02)) {
                goodMatches.add(matches.get(i));
            }
        }
        opencv_core.DMatchVector good_matches_Verctor = new opencv_core.DMatchVector(goodMatches.size());
        for (int i = 0; i < goodMatches.size(); i++) {
            good_matches_Verctor.put(i, goodMatches.get(i));
        }

        Mat img_matches = new Mat();
        opencv_features2d.drawMatches(currentFrame, surfKeyPoints, previousFrame, previousKeyPoint, good_matches_Verctor, img_matches);
        input.copyTo(previousFrame);
        previousKeyPoint.put(surfKeyPoints);
        currentFrameDescriptors.copyTo(previousFrameDescriptors);
        log("Finished matching");
        return img_matches;
    /*
        log("Start Grabcut ... ");
        Mat bgModel = new Mat(new Size(65, 1), opencv_core.CV_64FC1, opencv_core.Scalar.all(0));
        Mat fgModel = new Mat(new Size(65, 1), opencv_core.CV_64FC1, opencv_core.Scalar.all(0));
        Mat mask = new Mat();
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);
        backGroundModel.copyTo(mask);
        /*opencv_imgproc.grabCut(input, mask,
                new opencv_core.Rect(20, 20, input.cols() - 20, input.rows() - 20),
                bgModel, fgModel, 3,
                opencv_imgproc.GC_INIT_WITH_MASK);
        log("Stop Grabcut ... ");
        log("Time :" + (System.currentTimeMillis() - start));
        input.copyTo(previousFrame);
        previousKeyPoint.put(surfKeyPoints);
        currentFrameDescriptors.copyTo(previousFrameDescriptors);
        return mergeImageAndMask(copyOfOriginal, mask);
        */
    }

    private Mat mergeImageAndMask(Mat image, Mat mask) {
        Mat newImg = new Mat();
        image.copyTo(newImg);
        for (int y = 0; y < newImg.rows(); y++)
            for (int x = 0; x < newImg.cols(); x++) {
                for (int c = 0; c < newImg.channels(); c++) {
                    byte maskLabel = mask.ptr(y, x).get(0);
                    if (maskLabel == 2 || maskLabel == 0)
                        newImg.ptr(y, x).put(c, (byte) 0);
                }
            }
        return newImg;
    }

    public static byte[] float2ByteArray(float value) {
        byte[] array = ByteBuffer.allocate(4).putFloat(value).order(ByteOrder.LITTLE_ENDIAN).array();
        byte[] result = new byte[4];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[array.length - i - 1];
        }
        return result;
    }

    private void log(Object o) {
        System.out.println(o);
    }
}
