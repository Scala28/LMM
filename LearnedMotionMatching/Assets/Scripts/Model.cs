using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;

public class Model
{
    public float[] Mean_in { get; private set; }
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
    public void AddLayer(int inputSize, int outputSize, float[][] weight, float[] biases)
    {
        Layer l = new Layer(inputSize, outputSize);
        l.Weights = weight;
        l.Biases = biases;
        Layers.Add(l);
    }
    public void evaluate(float[] input, out float[] output)
    {
        float[] _in;
        float[] _out = new float[Layers[0].OutputSize];
        _in = input;
        nnLayer_normalize(input);
        for(int i=0; i<Layers.Count; i++)
        {
            if(i!=0)
                _out = new float[Layers[i].OutputSize];

            Layers[i].nnet_layer_linear(_in, _out);

            if (i != Layers.Count - 1)
            {
                Layers[i].nnet_layer_relu(_out);
                _in = _out;
            }
        }
        nnLayer_denormalize(_out);
        output = _out;
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
    public float[][] Weights;
    public float[] Biases;

    public Layer(int inputSize, int outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        Weights = new float[inputSize][];
        for(int i=0; i<inputSize; i++)
        {
            Weights[i] = new float[outputSize];
        }
        Biases = new float[outputSize];
    }
    public void nnet_layer_linear(float[] _in, float[] _out)
    {
        for(int j=0; j<_out.Length; j++)
        {
            _out[j]  = Biases[j];
        }
        for(int i=0; i<_in.Length; i++)
        {
            if (_in[i] != 0.0f)
                for(int j=0; j<_out.Length; j++)
                {
                    _out[j] += _in[i] * Weights[i][j];
                }
        }
    }
    public void nnet_layer_relu(float[] _out)
    {
        for(int i=0; i<_out.Length; i++)
            _out[i] = _out[i] > 0.0f ? _out[i] : 0;
    }
}
