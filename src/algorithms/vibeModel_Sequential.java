package algorithms;

import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.Random;

public class vibeModel_Sequential {
    /* Parameters. */
    private int width;
    private int height;
    private int numberOfSamples;


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

    public vibeModel_Sequential() {

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
        Random rand = new Random();
        this.updateFactor = updateFactor;
        int size = (this.width > this.height) ? 2 * this.width + 1 : 2 * this.height + 1;
        jump = new int[size];
        for (int i = 0; i < size; ++i) {
            this.jump[i] = (updateFactor == 1) ? 1 : (rand.nextInt() % (2 * this.updateFactor)) + 1;
        }
    }

    public void libvibeModel_Sequential_AllocInit_8u_C3R(Mat image_data){
        
    }
}
