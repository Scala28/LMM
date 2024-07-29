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
                int weightCols = reader.ReadInt32();
                int weightRows = reader.ReadInt32();
                float[] weightData = readFloat_toArray(reader, weightRows * weightCols);
                float[][] weights = new float[weightRows][];

                int biasLen = reader.ReadInt32();
                float[] biasData = readFloat_toArray(reader, biasLen);

                for (int row = 0; row < weightRows; row++)
                {
                    weights[row] = new float[weightCols];
                    for (int col = 0; col < weightCols; col++)
                    {
                        weights[row][col] = weightData[col * weightRows + row];
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
    private static Vector3[] readVec3_toArray(BinaryReader reader, int count)
    {
        byte[] buffer = reader.ReadBytes(count * 3 * sizeof(float));
        float[] temp = new float[count * 3];
        Vector3[] array = new Vector3[count];
        Buffer.BlockCopy(buffer, 0, temp, 0, buffer.Length);
        for(int i = 0; i < count; i++)
        {
            array[i].x = temp[i * 3];
            array[i].y = temp[i*3 + 1];
            array[i].z = temp[i*3 + 2];
        }
        return array;
    }
    private static Vector2[] readVec2_toArray(BinaryReader reader, int count)
    {
        byte[] buffer = reader.ReadBytes(count * 2 * sizeof(float));
        float[] temp = new float[count * 2];
        Vector2[] array = new Vector2[count];
        Buffer.BlockCopy(buffer, 0, temp, 0, buffer.Length);
        for(int i = 0; i < count; i++)
        {
            array[i].x = temp[i * 2];
            array[i].y = temp[i*2 + 1];
        }
        return array;
    }
    private static Vector4[] readVec4_toArray(BinaryReader reader, int count)
    {
        byte[] buffer = reader.ReadBytes(count * 4 * sizeof(float));
        float[] temp = new float[count * 4];
        Vector4[] array = new Vector4[count];
        Buffer.BlockCopy(buffer, 0, temp, 0, buffer.Length);
        for(int i = 0; i < count; i++)
        {
            array[i].x = temp[i * 4];
            array[i].y = temp[i*4 + 1];
            array[i].z = temp[i*4 + 2];
            array[i].w = temp[i*4 + 3];
        }
        return array;
    }
    private static short[] readShort_toArray(BinaryReader reader, int count)
    {
        byte[] buffer = reader.ReadBytes(count *  sizeof(short));
        short[] array = new short[count];
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
    private static short[][] readShort_toArray2d(BinaryReader reader, int rows, int cols)
    {
        byte[] buffer = reader.ReadBytes(rows * cols * sizeof (short));
        short[][] array2d = new short[rows][];
        for (int i = 0; i < rows; i++)
        {
            array2d[i] = new short[cols];
            Buffer.BlockCopy(buffer, i * cols * sizeof(short), array2d[i], 0, cols * sizeof(short));
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
    public static character load_character(string filename)
    {
        character c = new character();
        using (FileStream fs = new FileStream(filename, FileMode.Open, FileAccess.Read))
        using(BinaryReader reader = new BinaryReader(fs))
        {
            int count = reader.ReadInt32();
            c.positions = readVec3_toArray(reader, count);

            count = reader.ReadInt32();
            c.normals = readVec3_toArray(reader, count);

            count = reader.ReadInt32();
            c.texcoords = readVec2_toArray(reader, count);

            count = reader.ReadInt32();
            c.triangles = readShort_toArray(reader, count);

            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            c.bone_weights = readFloat_toArray2d(reader,rows,cols);

            rows = reader.ReadInt32();
            cols = reader.ReadInt32();
            c.bone_indices = readShort_toArray2d(reader, rows, cols);

            count = reader.ReadInt32();
            c.bone_rest_positions = readVec3_toArray(reader, count);

            count = reader.ReadInt32();
            c.bone_rest_rotations = readVec4_toArray(reader, count);
        }
        return c;
    }
    public static mesh gen_mesh_from_character(character c)
    {
        mesh mesh = new mesh();

        mesh.vertexCount = c.positions.Length;
        mesh.triangleCount = c.triangles.Length / 3;

        mesh.vertices = new float[c.positions.Length * 3];
        mesh.texcoords = new float[c.texcoords.Length * 2];
        mesh.normals = new float[c.normals.Length * 3];
        mesh.indices = new short[c.triangles.Length];

        for(int i=0; i<mesh.vertexCount; i++)
        {
            Vector3 pos = c.positions[i];
            mesh.vertices[i*3] = pos.x;
            mesh.vertices[i*3+1] = pos.y;
            mesh.vertices[i*3+2] = pos.z;
        }
        for(int i=0; i<c.texcoords.Length; i++)
        {
            Vector2 coord = c.texcoords[i];
            mesh.texcoords[i*2] = coord.x;
            mesh.texcoords[i*2+1] = coord.y;
        }
        for(int i=0; i<c.normals.Length; i++)
        {
            Vector3 normal = c.normals[i];
            mesh.normals[i*3] = normal.x;
            mesh.normals[i*3+1] = normal.y;
            mesh.normals[i*3+2] = normal.z;
        }
        for(int i=0; i<c.triangles.Length; i++)
        {
            mesh.indices[i] = c.triangles[i];
        }

        return mesh;
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
    public struct character
    {
        public Vector3[] positions;
        public Vector3[] normals;
        public Vector2[] texcoords;
        public short[] triangles;

        public float[][] bone_weights;
        public short[][] bone_indices;

        public Vector3[] bone_rest_positions;
        public Vector4[] bone_rest_rotations;

        public int nbones() { return this.bone_rest_positions.Length; }

        public static void liner_blend_skinning_positions(character c, Pose pose, out Vector3[] anim_positions)
        {
            anim_positions = new Vector3[c.nbones()];

            for(int i=0; i<c.nbones(); i++)
            {
                for(int j=0; j < c.bone_indices[0].Length; j++)
                {
                    if (c.bone_weights[i][j] > 0.0f)
                    {
                        int b = c.bone_indices[i][j];

                        Vector3 position = c.positions[i];
                        position = Quat.quat_mul_vec(Quat.quat_inv(c.bone_rest_rotations[b]),
                            position - c.bone_rest_positions[b]);
                        if (b == 0)
                            position = Quat.quat_mul_vec(pose.root_rotation, position) + pose.root_position;
                        else
                            position = Quat.quat_mul_vec(pose.joints[b-1].rotation, position) + pose.joints[b-1].position;

                        anim_positions[i] = anim_positions[i] + c.bone_weights[i][j] * position;
                    }
                }
            }
        }
        public static void liner_blend_skinning_normals(character c, Pose pose, out Vector3[] anim_normals)
        {
            anim_normals = new Vector3[c.nbones()];

            for(int i=0; i<c.nbones(); i++)
            {
                for(int j=0; j < c.bone_indices[0].Length; j++)
                {
                    if (c.bone_weights[i][j] > 0.0f)
                    {
                        int b = c.bone_indices[i][j];

                        Vector3 normal = c.normals[i];
                        normal = Quat.quat_mul_vec(Quat.quat_inv(c.bone_rest_rotations[b]), normal);
                        if (b == 0)
                            normal = Quat.quat_mul_vec(pose.root_rotation, normal);
                        else
                            normal = Quat.quat_mul_vec(pose.joints[b - 1].rotation, normal);

                        anim_normals[i] = anim_normals[i] + c.bone_weights[i][j] * normal;
                    }
                }
            }
            for(int i=0; i<anim_normals.Length; i++)
                anim_normals[i] = Quat.vec_normalize(anim_normals[i]);
        }
    }
    public struct mesh
    {
        public int vertexCount;
        public int triangleCount;

        public float[] vertices;
        public float[] texcoords;
        public float[] normals;
        public short[] indices;
    }
    private static int index(int bone, int vector, int component, int subcomponent, TensorShape shape)
    {
        return bone * shape.height * shape.width * shape.channels +
               vector * shape.width * shape.channels +
               component * shape.channels +
               subcomponent;
    }
}
