package algorithms;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.Random;

public class vibeModel_Sequential {
    /* Parameters. */
    private int width;
    private int height;
    private int numberOfSamples;
    private static int NUMBER_OF_HISTORY_IMAGES = 2;

    private int matchingThreshold;
    private int matchingNumber;
    private int updateFactor;

    /* Storage for the history. */
    private ArrayList<Mat> historyImage;
    private ArrayList<Mat> historyBuffer;
    private int lastHistoryImageSwapped;

    /* Buffers with random values. */
    private int[] jump;
    private int[] neighbor;
    private int[] position;
    Random rand;


    public vibeModel_Sequential() {
        rand = new Random();
        this.numberOfSamples = 20;
        this.matchingThreshold = 20;
        this.matchingNumber = 2;
        this.updateFactor = 16;

        this.historyImage = new ArrayList<>();
        this.historyBuffer = new ArrayList<>();
        this.lastHistoryImageSwapped = 0;
    }

    public int getNumberOfSamples() {
        return numberOfSamples;
    }

    public int getMatchingThreshold() {
        return matchingThreshold;
    }

    public int getMatchingNumber() {
        return matchingNumber;
    }

    public int getUpdateFactor() {
        return updateFactor;
    }


    public void setMatchingNumber(int matchingNumber) {
        this.matchingNumber = matchingNumber;
    }

    public void setUpdateFactor(int updateFactor) {

        this.updateFactor = updateFactor;
        int size = (this.width > this.height) ? 2 * this.width + 1 : 2 * this.height + 1;
        jump = new int[size];
        for (int i = 0; i < size; ++i) {
            this.jump[i] = (updateFactor == 1) ? 1 : (rand.nextInt() % (2 * this.updateFactor)) + 1;
        }
    }

    public void libvibeModel_Sequential_AllocInit_8u_C3R(Mat image_data) {
        this.width = width;
        this.height = height;
        for (int i = 0; i < NUMBER_OF_HISTORY_IMAGES; i++) {
            this.historyImage.add(new Mat(height, width, CvType.CV_8UC3));
        }

        for (int i = 0; i < NUMBER_OF_HISTORY_IMAGES; ++i) {
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++) {
                    historyImage.get(i).get(y, x)[0] = image_data.get(y, x)[0];
                    historyImage.get(i).get(y, x)[1] = image_data.get(y, x)[1];
                    historyImage.get(i).get(y, x)[2] = image_data.get(y, x)[2];
                }
        }

        for (int i = 0; i < numberOfSamples - NUMBER_OF_HISTORY_IMAGES; i++) {
            this.historyBuffer.add(new Mat(height, width, CvType.CV_8UC3));
        }

        /* Fills the history buffer */
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                int value_C1 = (int) image_data.get(y, x)[0];
                int value_C2 = (int) image_data.get(y, x)[1];
                int value_C3 = (int) image_data.get(y, x)[2];
                /* Fills the history buffer */
                for (int i = 0; i < historyBuffer.size(); ++i) {
                /* Adds noise on the value */
                    int value_plus_noise_C1 = value_C1 + rand.nextInt() % 20 - 10;
                    int value_plus_noise_C2 = value_C2 + rand.nextInt() % 20 - 10;
                    int value_plus_noise_C3 = value_C3 + rand.nextInt() % 20 - 10;

                /* Limits the value + noise to the [0,255] range */
                    if (value_plus_noise_C1 < 0) {
                        value_plus_noise_C1 = 0;
                    }
                    if (value_plus_noise_C1 > 255) {
                        value_plus_noise_C1 = 255;
                    }
                    if (value_plus_noise_C2 < 0) {
                        value_plus_noise_C2 = 0;
                    }
                    if (value_plus_noise_C2 > 255) {
                        value_plus_noise_C2 = 255;
                    }
                    if (value_plus_noise_C3 < 0) {
                        value_plus_noise_C3 = 0;
                    }
                    if (value_plus_noise_C3 > 255) {
                        value_plus_noise_C3 = 255;
                    }

                    historyBuffer.get(i).get(y, x)[0] = value_plus_noise_C1;
                    historyBuffer.get(i).get(y, x)[1] = value_plus_noise_C2;
                    historyBuffer.get(i).get(y, x)[2] = value_plus_noise_C3;
                }
            }

        /* Fills the buffers with random values. */
        int size = (width > height) ? 2 * width + 1 : 2 * height + 1;

        this.jump = new int[size];
        this.neighbor = new int[size];
        this.position = new int[size];

        for (int i = 0; i < size; ++i) {
            this.jump[i] = (rand.nextInt() % (2 * updateFactor)) + 1;            // Values between 1 and 2 * updateFactor.
            this.neighbor[i] = ((rand.nextInt() % 3) - 1) + ((rand.nextInt() % 3) - 1) * width; // Values between { width - 1, ... , width + 1 }.
            this.position[i] = rand.nextInt() % (numberOfSamples);               // Values between 0 and numberOfSamples - 1.
        }
    }

    public void libvibeModel_Sequential_Segmentation_8u_C3R(Mat image_data, Mat segmentation_map) {
        /* Segmentation. */
        for (int y = 0; y < image_data.height(); y++)
            for (int x = 0; x < image_data.width(); x++) {
                segmentation_map.get(y, x)[0] = matchingNumber - 1;
            }
        Mat first = historyImage.get(0);

        for (int y = 0; y < image_data.height(); y++)
            for (int x = 0; x < image_data.width(); x++) {
                double[] pixel = image_data.get(y, x);
                double[] hisPixel = first.get(y, x);
                if (!distance_is_close_8u_C3R((int) pixel[0], (int) pixel[1], (int) pixel[2],
                        (int) hisPixel[0], (int) hisPixel[1], (int) hisPixel[2], matchingThreshold)) {
                    segmentation_map.get(y, x)[0] = matchingNumber;
                }

            }

        for (int i = 1; i < NUMBER_OF_HISTORY_IMAGES; ++i) {
            Mat hisImg = historyImage.get(i);
            for (int y = 0; y < image_data.height(); y++)
                for (int x = 0; x < image_data.width(); x++) {
                    double[] pixel = image_data.get(y, x);
                    double[] hisPixel = hisImg.get(y, x);
                    if (distance_is_close_8u_C3R((int) pixel[0], (int) pixel[1], (int) pixel[2],
                            (int) hisPixel[0], (int) hisPixel[1], (int) hisPixel[2], matchingThreshold)) {
                        --segmentation_map.get(y, x)[0];
                    }
                }
        }

        // For swapping
        this.lastHistoryImageSwapped = (this.lastHistoryImageSwapped + 1) % NUMBER_OF_HISTORY_IMAGES;
        Mat swappingImageBuffer = historyImage.get(lastHistoryImageSwapped);

        // Now, we move in the buffer and leave the historyImages

        for (int y = 0; y < image_data.height(); y++) {
            for (int x = 0; x < image_data.width(); x++) {
                if (segmentation_map.get(y, x)[0] > 0) {
                /* We need to check the full border and swap values with the first or second historyImage.
                 * We still need to find a match before we can stop our search.
                 */
                    for (int numberOfTests = 0; numberOfTests < historyBuffer.size(); numberOfTests++) {
                        if (distance_is_close_8u_C3R(
                                (int) image_data.get(y, x)[0],
                                (int) image_data.get(y, x)[1],
                                (int) image_data.get(y, x)[2],
                                (int) historyBuffer.get(numberOfTests).get(y, x)[0],
                                (int) historyBuffer.get(numberOfTests).get(y, x)[1],
                                (int) historyBuffer.get(numberOfTests).get(y, x)[2],
                                matchingThreshold)) {
                            --segmentation_map.get(y, x)[0];
                        }

                    /* Swaping: Putting found value in history image buffer. */
                        int temp_r = (int) swappingImageBuffer.get(y, x)[0];
                        int temp_g = (int) swappingImageBuffer.get(y, x)[1];
                        int temp_b = (int) swappingImageBuffer.get(y, x)[2];

                        swappingImageBuffer.get(y, x)[0] = historyBuffer.get(numberOfTests).get(y, x)[0];
                        swappingImageBuffer.get(y, x)[1] = historyBuffer.get(numberOfTests).get(y, x)[1];
                        swappingImageBuffer.get(y, x)[2] = historyBuffer.get(numberOfTests).get(y, x)[2];

                        historyBuffer.get(numberOfTests).get(y, x)[0] = temp_r;
                        historyBuffer.get(numberOfTests).get(y, x)[1] = temp_g;
                        historyBuffer.get(numberOfTests).get(y, x)[2] = temp_b;

                        if (segmentation_map.get(y, x)[0] <= 0) break;
                    }
                }
                if (segmentation_map.get(y, x)[0] > 0) {
                    segmentation_map.get(y, x)[0] = 255;
                }
            }
        }
    }

    public void libvibeModel_Sequential_Update_8u_C3R(Mat image_data, Mat updating_mask) {
         /* All the frame, except the border. */
        int shift, indX, indY;
        int x, y;

        for (y = 1; y < height - 1; ++y) {
            shift = rand.nextInt() % width;
            indX = jump[shift]; // index_jump should never be zero (> 1).

            while (indX < width - 1) {
                if (updating_mask.get(y, indX)[0] == 0) {
        /* In-place substitution. */
                    int r = (int) image_data.get(y, indX)[0];
                    int g = (int) image_data.get(y, indX)[1];
                    int b = (int) image_data.get(y, indX)[2];

                    int index_neighborX = (indX + neighbor[shift]);
                    int index_neighborY = y;
                    if (index_neighborX > width) {
                        index_neighborX %= width;
                        index_neighborY++;
                    }

                    if (position[shift] < NUMBER_OF_HISTORY_IMAGES) {
                        historyImage.get(position[shift]).get(y, indX)[0] = r;
                        historyImage.get(position[shift]).get(y, indX)[1] = g;
                        historyImage.get(position[shift]).get(y, indX)[2] = b;

                        historyImage.get(position[shift]).get(index_neighborY, index_neighborX)[0] = r;
                        historyImage.get(position[shift]).get(index_neighborY, index_neighborX)[1] = g;
                        historyImage.get(position[shift]).get(index_neighborY, index_neighborX)[2] = b;
                    } else {
                        int pos = position[shift] - NUMBER_OF_HISTORY_IMAGES;

                        historyBuffer.get(pos).get(y, indX)[0] = r;
                        historyBuffer.get(pos).get(y, indX)[1] = g;
                        historyBuffer.get(pos).get(y, indX)[2] = b;

                        historyBuffer.get(pos).get(index_neighborY, index_neighborX)[0] = r;
                        historyBuffer.get(pos).get(index_neighborY, index_neighborX)[1] = g;
                        historyBuffer.get(pos).get(index_neighborY, index_neighborX)[2] = b;
                    }
                }
                ++shift;
                indX += jump[shift];
            }
        }
        //TODO debug + check first row , first column , first pixel if need
    }

    private boolean distance_is_close_8u_C3R(int r1, int g1, int b1, int r2, int g2, int b2, int threshold) {
        return (abs_uint(r1 - r2) + abs_uint(g1 - g2) + abs_uint(b1 - b2) <= 4.5 * threshold);
    }

    private int abs_uint(int i) {
        return (i >= 0) ? i : -i;
    }

}
