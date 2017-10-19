package algorithms;

import org.opencv.core.Mat;

import java.util.ArrayList;

public class KernelDensityEstimator {
    private ArrayList<Mat> historyMat = new ArrayList<>();
    private int dirtyIndex = 0;
    private double N = 0;
    private double threshold = 0.5;

    public void setN(int n) {
        this.N = n;
    }

    public Mat foregroundMask(Mat currentFrame) throws KernelDensityEstimatorException {
        Mat foreGround = new Mat(currentFrame.rows(), currentFrame.cols(), currentFrame.type());
        if (currentFrame.channels() != 3)
            throw new KernelDensityEstimatorException("Frame chanel muss be 3!");
        int width = currentFrame.width();
        int height = currentFrame.height();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double[] pixel = currentFrame.get(x, y);
                double kdeValue = 0;
                for (int i = 0; i < historyMat.size(); i++) {
                    double[] historyPixel = historyMat.get(i).get(x, y);
                    kdeValue += factor(pixel[0], historyPixel[0])
                            * factor(pixel[1], historyPixel[1])
                            * factor(pixel[2], historyPixel[2]);
                }
                kdeValue /= historyMat.size();
                if (kdeValue < threshold) {
                    foreGround.put(x, y, new double[]{1, 1, 1});
                } else {
                    foreGround.put(x, y, new double[]{0, 0, 0});
                }
            }
        }
        return foreGround;
    }

    private double factor(double current, double history) {
        double firstTerm = Math.sqrt(2 * Math.PI * kernelFunction(0));
        double secondTerm = -0.5d * Math.pow((current - history), 2);
        return 1 / Math.sqrt(firstTerm) * secondTerm;
    }

    private void updateHistoryMat(Mat mat) {
        if (historyMat.size() >= N) {
            historyMat.set(dirtyIndex, mat);
            dirtyIndex = (dirtyIndex++) % (int) Math.round(N);
        }
    }

    private double kernelFunction(double medianAbsolutDeviation) {
        return medianAbsolutDeviation / (0.68d * Math.sqrt(2.0d));
    }


    public class KernelDensityEstimatorException extends Exception {
        public KernelDensityEstimatorException() {
            super();
        }

        public KernelDensityEstimatorException(String message) {
            super(message);
        }

        public KernelDensityEstimatorException(String message, Throwable cause) {
            super(message, cause);
        }

        public KernelDensityEstimatorException(Throwable cause) {
            super(cause);
        }
    }
}
