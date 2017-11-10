package mainCpp;

import org.bytedeco.javacpp.opencv_core;

public class KeyPointsAndFeaturesVector {
    opencv_core.KeyPointVector keypointVector;
    opencv_core.Mat descriptors;

    public KeyPointsAndFeaturesVector() {
        this.keypointVector = new opencv_core.KeyPointVector();
        this.descriptors = new opencv_core.Mat();
    }

    public KeyPointsAndFeaturesVector(opencv_core.KeyPointVector keypointVector, opencv_core.Mat descriptors) {
        this.keypointVector = keypointVector;
        this.descriptors = descriptors;
    }

    public void addNewKeyPointAndDescriptors(opencv_core.KeyPoint keyPoint, opencv_core.Mat descriptor) throws KeyPointsAndFeaturesVectorException {
        if (descriptor.cols() == descriptors.cols()) {
            keypointVector.resize(keypointVector.size() + 1);
            keypointVector.put(keypointVector.size() - 1, keyPoint);
            descriptors.push_back(descriptor);
        } else
            throw new KeyPointsAndFeaturesVectorException("new col:" + descriptor.cols() + " is not equal " + +descriptors.cols());
    }

    public opencv_core.KeyPoint getKeypoint(int index) {
        return keypointVector.get(index);
    }

    public opencv_core.Mat getDescriptor(int index) {
        return descriptors.rows(index);
    }

    public class KeyPointsAndFeaturesVectorException extends Exception {
        public KeyPointsAndFeaturesVectorException(String msg) {
            super(msg);
        }
    }
}
