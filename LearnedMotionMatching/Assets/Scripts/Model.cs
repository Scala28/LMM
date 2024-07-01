using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Model
{
    public float[] Mean_in {  get; private set; }
    public float[] Mean_out { get; private set; }
    public float[] Std_in { get; private set; }
    public float[] Std_out { get; private set; }

    public Model(float[] mean_in, float[] mean_out, float[] std_in, float[] std_out)
    {
        Mean_in = mean_in;
        Mean_out = mean_out;
        Std_in = std_in;
        Std_out = std_out;
        Layers = new List<Layer>();
    }

    public List<Layer> Layers { get; private set; }

    public void AddLayer(Layer l) { Layers.Add(l); }
    public void AddLayer(int inputSize, int outputSize, float[,] weight, float[] biases)
    {
        Layer l = new Layer(inputSize, outputSize);
        l.Weights = weight;
        l.Biases = biases;
        Layers.Add(l);
    }
}
public class Layer
{
    public int InputSize;
    public int OutputSize;
    public float[,] Weights;
    public float[] Biases;

    public Layer(int inputSize, int outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        Weights = new float[inputSize, outputSize];
        Biases = new float[outputSize];
    }
}
