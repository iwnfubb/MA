package algorithms;

import org.opencv.core.*;

import java.util.ArrayList;
import java.util.Arrays;

public class KernelDensityEstimator {
    private ArrayList<Mat> historyMat = new ArrayList<>();
    private int dirtyIndex = 0;
    private double N = 1;
    private double threshold = 0.1;
    private int width = 0;
    private int height = 0;

    public void setN(int n) {
        this.N = n;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    public Mat foregroundMask(Mat currentFrame) throws KernelDensityEstimatorException {
        Mat foreGround = new Mat(currentFrame.rows(), currentFrame.cols(), currentFrame.type());
        if (currentFrame.channels() != 3) {
            throw new KernelDensityEstimatorException("Frame chanel muss be 3!");
        }
        if (historyMat.size() != 0) {
            this.height = currentFrame.rows();
            this.width = currentFrame.cols();
            startTimeCounter();
            double[][] deep = new double[3][historyMat.size()];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double[] pixel = currentFrame.get(y, x);
                    double kdeValue = 0;
                    for (int i = 0; i < historyMat.size(); i++) {
                        double[] pixel_His = historyMat.get(i).get(y, x);
                        deep[0][i] = pixel_His[0];
                        deep[1][i] = pixel_His[1];
                        deep[2][i] = pixel_His[2];
                    }
                    for (int i = 0; i < historyMat.size(); i++) {
                        double[] historyPixel = historyMat.get(i).get(y, x);
                        kdeValue += factor(pixel[0], historyPixel[0], kernelFunction(mad(deep[0])))
                                * factor(pixel[1], historyPixel[1], kernelFunction(mad(deep[1])))
                                * factor(pixel[2], historyPixel[2], kernelFunction(mad(deep[2])));
                    }
                    kdeValue /= historyMat.size();
                    if (kdeValue < threshold) {
                        foreGround.put(y, x, new double[]{255, 255, 255});
                    } else {
                        foreGround.put(y, x, new double[]{0, 0, 0});
                    }
                }
            }
            stopTimeCounter("cal His");
        }
        updateHistoryMat(currentFrame);
        return foreGround;
    }

    private double factor(double chanel, double historyChanel, double madChanel) {
        double firstTerm = Math.sqrt(2 * Math.PI * kernelFunction(0));
        double secondTerm = -0.5d * Math.pow((chanel - historyChanel), 2);
        return 1 / Math.sqrt(firstTerm) * secondTerm;
    }

    private void updateHistoryMat(Mat mat) {
        if (historyMat.size() <= N) {
            historyMat.add(mat);
        } else if (historyMat.size() > N) {
            historyMat.set(dirtyIndex, mat);
            dirtyIndex = (dirtyIndex++) % historyMat.size();
        }
    }

    private double kernelFunction(double medianAbsolutDeviation) {
        return medianAbsolutDeviation / (0.68d * Math.sqrt(2.0d));
    }

    private Double mad(double[] input) {
        double median = median(input);
        arrayAbsDistance(input, median);
        return median(input);
    }

    private static void arrayAbsDistance(double[] array, double value) {
        for (int i = 0; i < array.length; i++) {
            array[i] = Math.abs(array[i] - value);
        }
    }

    private static double median(double[] input) {
        if (input.length == 0) {
            throw new IllegalArgumentException("to calculate median we need at least 1 element");
        }
        Arrays.sort(input);
        if (input.length % 2 == 0) {
            return (input[input.length / 2 - 1] + input[input.length / 2]) / 2;
        }
        return input[input.length / 2];
    }

    public class KernelDensityEstimatorException extends Exception {
        public KernelDensityEstimatorException(String message) {
            super(message);
        }
    }

    private long start = System.currentTimeMillis();

    private void startTimeCounter() {
        start = System.currentTimeMillis();
    }

    private void stopTimeCounter(String msg) {
        long stop = System.currentTimeMillis();
        System.out.println(msg + "Time: " + (stop - start));
    }

}
