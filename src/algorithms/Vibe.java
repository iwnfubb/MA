package algorithms;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class Vibe {
    /* Variables. */
    static int frameNumber = 1; /* The current frame number */
    Mat frame;                  /* Current frame. */
    Mat segmentationMap;        /* Will contain the segmentation map. This is the binary output map. */
    int keyboard = 0;           /* Input from keyboard. Used to stop the program. Enter 'q' to quit. */

    public Vibe(){

    }

    private void foregroundMask(Mat currentFrame){
        if (frameNumber == 1) {
            segmentationMap = new Mat(frame.rows(), frame.cols(), CvType.CV_8UC1);
            //model = (vibeModel_Sequential_t*)libvibeModel_Sequential_New();
            //libvibeModel_Sequential_AllocInit_8u_C3R(model, frame.data, frame.cols, frame.rows);
        }

    }

}
