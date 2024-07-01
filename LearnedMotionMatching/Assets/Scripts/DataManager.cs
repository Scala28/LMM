using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEditor;
using System;


public static class DataManager
{
    // Read .bin file from Resources folder
    public static (int, int, float[]) Load_database_fromResources(string filename)
    {
        TextAsset binAsset = Resources.Load(filename) as TextAsset; 
        if(binAsset == null)
        {
            Debug.Log("Failed to load .bin file " + filename);
            return (0, 0, null);
        }

        using (MemoryStream memStream = new MemoryStream(binAsset.bytes))
        using (BinaryReader reader = new BinaryReader(memStream))
        {
            int nframes = reader.ReadInt32();
            int ndata = reader.ReadInt32();
            float[] data = new float[nframes * ndata];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = reader.ReadSingle();
            }
            return (nframes, ndata, data);
        }
    }
    public static Model Load_net_fromParameters(string filename)
    {
        using (FileStream fs = new FileStream(filename, FileMode.Open, FileAccess.Read))
        using (BinaryReader reader = new BinaryReader(fs))
        {
            int meanInLen = reader.ReadInt32();
            float[] meanIn = ReadFloatArray(reader, meanInLen);

            int stdInLen = reader.ReadInt32();
            float[] stdIn = ReadFloatArray(reader, stdInLen);

            int meanOutLen = reader.ReadInt32();
            float[] meanOut = ReadFloatArray(reader, meanOutLen);

            int stdOutLen = reader.ReadInt32();
            float[] stdOut = ReadFloatArray(reader, stdOutLen);

            Model model = new Model(meanIn, meanOut, stdIn, stdOut);

            int numLayers = reader.ReadInt32();

            for(int i = 0; i < numLayers; i++)
            {
                int weightCols = reader.ReadInt32();
                int weightRows = reader.ReadInt32();
                float[] weightData = ReadFloatArray(reader, weightRows * weightCols);
                float[,] weights = new float[weightRows, weightCols];

                int biasLen = reader.ReadInt32();
                float[] biasData = ReadFloatArray(reader, biasLen);

                for (int row = 0; row < weightRows; row++)
                {
                    for (int col = 0; col < weightCols; col++)
                    {
                        weights[row, col] = weightData[row * weightCols + col];
                    }
                }
                model.AddLayer(weightRows, weightCols, weights, biasData);
            }
            return model;
        }
    }   
    private static float[] ReadFloatArray(BinaryReader reader, int count)
    {
        byte[] buffer = reader.ReadBytes(count * sizeof(float));
        float[] array = new float[count];
        Buffer.BlockCopy(buffer, 0, array, 0, buffer.Length);
        return array;
    }
}
