import org.omg.CORBA.DATA_CONVERSION;

import java.util.ArrayList;
import java.util.Arrays;

public class Perceptron {
    private int numInputs;
    private float[] weights;
    private float THRESHOLD = 0;

    private String classifyForLabel; // this is a classifier for eg "virginica"
    private float learningRate = 0.01f;

    public Perceptron(int numInputs, String whatToClassify) {
        this.classifyForLabel = whatToClassify;
        this.numInputs = numInputs;
        weights = initWeights(numInputs);
    }

    private float[] initWeights(int numInputs) {
        float[] out = new float[numInputs];
        for (int i = 0; i < numInputs; i++) {
            out[i] = -0.5f;
        }
        return out;
    }

    /***
     * Train the perceptron using the input feature vector and its correct label.
     * Return true if there was a non-zero error and training occured (weights got adjusted)
     *
     * @param batch
     * @param features
     * @return
     */
    public boolean train(ArrayList<DataSet.DataPoint> batch, String[] features) {
        float[] weightUpdates = new float[features.length];
        float thresholdUpdate = 0;

        for (DataSet.DataPoint point : batch) {
            float[] input = point.getData(features);
            String correctLabel = point.getLabelString();

            float prediction = guess(input);
            int correctAnswer = getCorrectGuess(correctLabel);
            float error = prediction - correctAnswer;

            if (prediction != correctAnswer) {
                for (int i = 0; i < weights.length; i++) {
                    weightUpdates[i] = input[i] * error * prediction * (1 - prediction) * learningRate;
                }

                thresholdUpdate += error * prediction * (1 - prediction) * learningRate;
            }
        }

        for (int i = 0; i < weights.length; i++) {
            weights[i] -= weightUpdates[i];
        }
        THRESHOLD -= thresholdUpdate;

        return true;

    }

    public float guess(float[] input) {
        float sum = 0;
        for (int i = 0; i < input.length; i++) {
            sum += input[i] * weights[i];
        }

        return activationFunction(sum);
    }

    private float activationFunction(float sum) {
        return (float) (1.0 / (1 + Math.exp(-sum)));
    }

    public float[] getWeights() {
        return weights;
    }

    public String getTargetLabel() {
        return this.classifyForLabel;
    }

    public boolean isGuessCorrect(int guess, String correctLabel) {
        return guess == getCorrectGuess(correctLabel);
    }

    /***
     * Return the correct output for a given class label.  ie returns 1 if input label matches
     * what this perceptron is trying to classify.
     * @param label
     * @return
     */
    public int getCorrectGuess(String label) {
        if (label.equals(this.classifyForLabel))
            return 1;
        else
            return 0;
    }

    public float getThreshold() {
        return THRESHOLD;
    }
}