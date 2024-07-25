using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
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

    public void evaluate(float[] input, ref float[] output)
    {
        nnLayer_normalize(input);
        for(int i=0; i < Layers.Count-1; i++)
        {
            float[] _out = Layers[i].linear(input);
            Layer.relu(_out);
            input = _out; ;
        }
        output = Layers[Layers.Count - 1].linear(input);
        nnLayer_denormalize(output);
    }
    private void nnLayer_denormalize(float[] _out)
    {
        for (int i = 0; i < Mean_out.Length; i++)
        {
            _out[i] = _out[i] * Std_out[i] + Mean_out[i];
        }

    }
    private void nnLayer_normalize(float[] _out)
    {
        for (int i = 0; i < Mean_in.Length; i++)
        {
            _out[i] = (_out[i] - Mean_in[i]) / Std_in[i];
        }
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
    public float[] linear(float[] input)
    {
        float[] output = new float[OutputSize];
        for(int j=0; j<output.Length; j++)
        {
            output[j] = Biases[j];
        }

        for(int i=0; i<input.Length; i++)
        {
            if (input[i] != 0.0f)
                for (int j = 0; j < output.Length; j++)
                    output[j] += input[i] * Weights[i, j];
        }
        return output;
    }
    public static void relu(float[] output)
    {
        for(int i=0; i<output.Length; i++)
            output[i] = Mathf.Max(output[i], 0.0f);
    }
}
