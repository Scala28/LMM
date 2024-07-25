using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEditor;
using System;
using UnityEngine.Scripting;
using Unity.Barracuda;


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
            float[] meanIn = readFloat_toArray(reader, meanInLen);

            int stdInLen = reader.ReadInt32();
            float[] stdIn = readFloat_toArray(reader, stdInLen);

            int meanOutLen = reader.ReadInt32();
            float[] meanOut = readFloat_toArray(reader, meanOutLen);

            int stdOutLen = reader.ReadInt32();
            float[] stdOut = readFloat_toArray(reader, stdOutLen);

            Model model = new Model(meanIn, meanOut, stdIn, stdOut);

            int numLayers = reader.ReadInt32();

            for(int i = 0; i < numLayers; i++)
            {
                int weightRows = reader.ReadInt32();
                int weightCols = reader.ReadInt32();
                float[] weightData = readFloat_toArray(reader, weightRows * weightCols);
                float[,] weights = new float[weightRows, weightCols];

                int biasLen = reader.ReadInt32();
                float[] biasData = readFloat_toArray(reader, biasLen);

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
    private static float[] readFloat_toArray(BinaryReader reader, int count)
    {
        byte[] buffer = reader.ReadBytes(count * sizeof(float));
        float[] array = new float[count];
        Buffer.BlockCopy(buffer, 0, array, 0, buffer.Length);
        return array;
    }
    private static int[] readInt_toArray(BinaryReader reader, int count)
    {
        byte[] buffer = reader.ReadBytes(count * sizeof(int));
        int[] array = new int[count];
        Buffer.BlockCopy(buffer, 0, array, 0, buffer.Length);
        return array;
    }
    private static Vector3[][] readVec3_toArray2d(BinaryReader reader, int rows, int cols)
    {
        byte[] buffer = reader.ReadBytes(rows * cols * 3 * sizeof(float));
        Vector3[][] array2d = new Vector3[rows][];
        for(int i=0; i<rows; i++)
        {
            array2d[i] = new Vector3[cols];
            for(int j=0; j < cols; j++)
            {
                Vector3 vec = new Vector3();
                float[] temp = new float[3];
                Buffer.BlockCopy(buffer, index(i, j, 0, 0, new TensorShape(rows, cols, 3, sizeof(float))) , temp, 0, 3 * sizeof(float));
                vec.x = temp[0];
                vec.y = temp[1];
                vec.z = temp[2];
                array2d[i][j] = vec;
            }
        }
        return array2d;
    }
    private static Vector4[][] readVec4_toArray2d(BinaryReader reader, int rows, int cols)
    {
        byte[] buffer = reader.ReadBytes(rows * cols * 4 * sizeof(float));
        Vector4[][] array2d = new Vector4[rows][];
        for (int i = 0; i < rows; i++)
        {
            array2d[i] = new Vector4[cols];
            for (int j = 0; j < cols; j++)
            {
                Vector4 vec = new Vector4();
                float[] temp = new float[4];
                Buffer.BlockCopy(buffer, index(i, j, 0, 0, new TensorShape(rows, cols, 4, sizeof(float))), temp, 0, 4 * sizeof(float));
                vec.x = temp[0];
                vec.y = temp[1];
                vec.z = temp[2];
                vec.w = temp[3];
                array2d[i][j] = vec;
            }
        }
        return array2d;
    }
    private static bool[][] readBool_toArray2d(BinaryReader reader, int rows, int cols)
    {
        byte[] buffer = reader.ReadBytes(rows * cols * sizeof(bool));
        bool[][] array2d = new bool[rows][];
        for (int i = 0; i < rows; i++)
        {
            array2d[i] = new bool[cols];
            Buffer.BlockCopy(buffer, i * cols * sizeof(bool), array2d[i], 0, cols * sizeof(bool));
        }
        return array2d;
    }
    private static float[][] readFloat_toArray2d(BinaryReader reader, int rows, int cols)
    {
        byte[] buffer = reader.ReadBytes(rows * cols * sizeof (float));
        float[][] array2d = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            array2d[i] = new float[cols];
            Buffer.BlockCopy(buffer, i * cols * sizeof(float), array2d[i], 0, cols * sizeof(float));
        }
        return array2d;
    }
    public static database load_database(string filename)
    {
        database db = new database();
        using(FileStream fs = new FileStream(filename, FileMode.Open, FileAccess.Read))
        using(BinaryReader reader = new BinaryReader(fs))
        {
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            db.bone_positions = readVec3_toArray2d(reader, rows, cols);

            rows = reader.ReadInt32(); 
            cols = reader.ReadInt32();
            db.bone_velocities = readVec3_toArray2d(reader, rows, cols);

            rows = reader.ReadInt32();
            cols = reader.ReadInt32();
            db.bone_rotations = readVec4_toArray2d(reader, rows, cols);

            rows = reader.ReadInt32();
            cols = reader.ReadInt32();
            db.bone_angular_velocities = readVec3_toArray2d(reader, rows, cols);

            int count = reader.ReadInt32();
            db.bone_parents = readInt_toArray(reader, count);

            count = reader.ReadInt32();
            db.range_starts = readInt_toArray(reader, count);

            count = reader.ReadInt32();
            db.range_stops = readInt_toArray(reader, count);

            rows = reader.ReadInt32();
            cols = reader.ReadInt32();
            db.contact_states = readBool_toArray2d(reader, rows, cols);
        }
        return db;
    }
    public static (float[][], float[], float[]) load_features(string filename)
    {
        float[][] features;
        float[] features_offset;
        float[] features_scale;
        using(FileStream fs = new FileStream(filename, FileMode.Open, FileAccess.Read))
        using(BinaryReader reader = new BinaryReader(fs))
        {
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            features = readFloat_toArray2d(reader, rows, cols );

            int count = reader.ReadInt32();
            features_offset = readFloat_toArray(reader, count);

            count = reader.ReadInt32();
            features_scale = readFloat_toArray(reader, count);
        }

        return (features, features_offset, features_scale);
    }
    public struct database
    {
        public Vector3[][] bone_positions;
        public Vector3[][] bone_velocities;
        public Vector4[][] bone_rotations;
        public Vector3[][] bone_angular_velocities;

        public int[] bone_parents;

        public int[] range_starts;
        public int[] range_stops;

        public float[][] features;
        public float[] features_offset;
        public float[] features_scale;

        public bool[][] contact_states;

        float[][] bound_sm_min;
        float[][] bound_sm_max;
        float[][] bound_lr_min;
        float[][] bound_lr_max;

        public int nframes() { return bone_positions.Length; }
        public int nbones() { return bone_positions[0].Length; }
        public int nfeatures() { return features[0].Length; }
    }
    private static int index(int bone, int vector, int component, int subcomponent, TensorShape shape)
    {
        return bone * shape.height * shape.width * shape.channels +
               vector * shape.width * shape.channels +
               component * shape.channels +
               subcomponent;
    }
}
