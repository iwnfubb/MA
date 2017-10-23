package algorithms;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class Vibe {
    /* Variables. */
    static int frameNumber = 1; /* The current frame number */
    Mat segmentationMap;        /* Will contain the segmentation map. This is the binary output map. */
    public VibeModel model;

    public Vibe() {
        this.model =  new VibeModel();
    }

    public Mat foregroundMask(Mat currentFrame) {
        if (frameNumber == 1) {
            segmentationMap = new Mat(currentFrame.rows(), currentFrame.cols(), CvType.CV_8UC1);
            model.vibeModel_Sequential_Init_8u_C3R(currentFrame);
        }else {
            model.vibeModel_Segmentation_8u_C3R(currentFrame, segmentationMap);
            model.vibeModel_Update_8u_C3R(currentFrame, segmentationMap);
        }
        frameNumber++;
        return segmentationMap;
    }

}
