using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEditor;
using System;
using Unity.Barracuda;


public static class DataManager
{
    // Maximum value of a float, from bit pattern 01111111011111111111111111111111
    private const float FLT_MAX = 340282346638528859811704183484516925440.0f;

    #region Build Matching features

    private const int BOUND_SM_SIZE = 16;
    private const int BOUND_LR_SIZE = 64;

    private static void normalize_features(float[][] features, float[] feature_offsets, float[] feature_scales, 
        int offset, int size, float weight = 1.0f)
    {
        for(int j=0; j<size; j++)
        {
            feature_offsets[offset + j] = 0.0f;
        }
        for(int i=0; i<features.Length; i++)
        {
            for(int j=0; j<size; j++)
            {
                feature_offsets[offset + j] += features[i][offset + j] / features.Length;
            }
        }

        float[] vars = new float[size];

        for(int i=0; i<features.Length; i++) { 
            for(int j=0; j<vars.Length; j++)
            {
                vars[j] += squaref(features[i][offset + j] - feature_offsets[offset + j]) / features.Length;
            }
        }

        float std = 0.0f;
        for(int j=0; j<size; j++)
        {
            std += Mathf.Sqrt(vars[j]) / size;
        }

        Debug.Assert(std > 0.0f);

        for(int j=0; j < size; j++)
        {
            feature_scales[offset + j] = std / weight;
        }

        for(int i=0; i<features.Length; i++)
        {
            for(int j=0; j<size; j++)
            {
                features[i][offset+j] = (features[i][offset+j] - feature_offsets[offset + j]) / feature_scales[offset + j];
            }
        }
    }
    private static void compute_bone_position_feature(ref database db, ref int offset, int bone, float weight = 1.0f)
    {
        for(int i=0; i<db.nframes();  i++)
        {
            Vector3 bone_position;
            Vector4 bone_rotation;

            forward_kinematics(out bone_position, out bone_rotation,
                db.bone_positions[i], db.bone_rotations[i], db.bone_parents, bone);

            bone_position = Quat.quat_mul_vec(Quat.quat_inv(db.bone_rotations[i][0]), bone_position - db.bone_positions[i][0]);

            db.features[i][offset + 0] = bone_position.x;
            db.features[i][offset + 1] = bone_position.y;
            db.features[i][offset + 2] = bone_position.z;
        }
        normalize_features(db.features, db.features_offset, db.features_scale, offset, 3, weight);

        offset += 3;
    }
    private static void compute_bone_height_feature(ref database db, ref int offset, int bone, float foot_height, float weight = 1.0f)
    {
        for (int i = 0; i < db.nframes(); i++)
        {
            int t1 = db.database_trajectory_index_clamp(i, 15);
            int t2 = db.database_trajectory_index_clamp(i, 30);
            int t3 = db.database_trajectory_index_clamp(i, 45);

            Vector3 t0_pos;
            Vector4 t0_rot;

            forward_kinematics(out t0_pos, out t0_rot,
                db.bone_positions[i], db.bone_rotations[i], db.bone_parents, bone);

            Vector3 t1_pos;
            Vector4 t1_rot;

            forward_kinematics(out t1_pos, out t1_rot,
                db.bone_positions[t1], db.bone_rotations[t1], db.bone_parents, bone);

            Vector3 t2_pos;
            Vector4 t2_rot;

            forward_kinematics(out t2_pos, out t2_rot,
                db.bone_positions[t2], db.bone_rotations[t2], db.bone_parents, bone);

            Vector3 t3_pos;
            Vector4 t3_rot;

            forward_kinematics(out t3_pos, out t3_rot,
                db.bone_positions[t3], db.bone_rotations[t3], db.bone_parents, bone);

            // This height is recorded relative to the
            // current height of the character’s root

            db.features[i][offset + 0] = t0_pos.y - foot_height - db.bone_positions[i][0].y;
            db.features[i][offset + 1] = t1_pos.y - foot_height - db.bone_positions[i][0].y;
            db.features[i][offset + 2] = t2_pos.y - foot_height - db.bone_positions[i][0].y;
            db.features[i][offset + 3] = t3_pos.y - foot_height - db.bone_positions[i][0].y;
        }

        normalize_features(db.features, db.features_offset, db.features_scale, offset, 4, weight);

        offset += 4;
    }
    private static void compute_bone_velocity_feature(ref database db, ref int offset, int bone, float weight = 1.0f)
    {
        for(int i=0; i<db.nframes();i++)
        {
            Vector3 bone_position;
            Vector4 bone_rotation;
            Vector3 bone_velocity;
            Vector3 bone_angular_velocity;

            forward_kinematics_velocity(out bone_position, out bone_rotation, out bone_velocity, out bone_angular_velocity,
                db.bone_positions[i], db.bone_rotations[i], db.bone_velocities[i], db.bone_angular_velocities[i], db.bone_parents, bone);

            bone_velocity = Quat.quat_mul_vec(Quat.quat_inv(db.bone_rotations[i][0]), bone_velocity);

            db.features[i][offset + 0] = bone_velocity.x;
            db.features[i][offset + 1] = bone_velocity.y;
            db.features[i][offset + 2] = bone_velocity.z;
        }

        normalize_features(db.features, db.features_offset, db.features_scale, offset, 3, weight);

        offset += 3;
    }
    private static void compute_trajectory_position_feature(ref database db, ref int offset, float weight = 1.0f)
    {
        for(int i=0; i<db.nframes(); i++)
        {
            int t0 = db.database_trajectory_index_clamp(i, 20);
            int t1 = db.database_trajectory_index_clamp(i, 40);
            int t2 = db.database_trajectory_index_clamp(i, 60);

            Vector3 trajectory_pos0 = Quat.quat_mul_vec(Quat.quat_inv(db.bone_rotations[i][0]),
                db.bone_positions[t0][0] - db.bone_positions[i][0]);
            Vector3 trajectory_pos1 = Quat.quat_mul_vec(Quat.quat_inv(db.bone_rotations[i][0]),
                db.bone_positions[t1][0] - db.bone_positions[i][0]);
            Vector3 trajectory_pos2 = Quat.quat_mul_vec(Quat.quat_inv(db.bone_rotations[i][0]),
                db.bone_positions[t2][0] - db.bone_positions[i][0]);

            db.features[i][offset + 0] = trajectory_pos0.x;
            db.features[i][offset + 1] = trajectory_pos0.z;
            db.features[i][offset + 2] = trajectory_pos1.x;
            db.features[i][offset + 3] = trajectory_pos1.z;
            db.features[i][offset + 4] = trajectory_pos2.x;
            db.features[i][offset + 5] = trajectory_pos2.z;
        }

        normalize_features(db.features, db.features_offset, db.features_scale, offset, 6, weight);

        offset += 6;
    }
    private static void compute_trajectory_direction_feature(ref database db, ref int offset, float weight = 1.0f)
    {
        for(int i=0; i<db.nframes(); i++)
        {
            int t0 = db.database_trajectory_index_clamp(i, 20);
            int t1 = db.database_trajectory_index_clamp(i, 40);
            int t2 = db.database_trajectory_index_clamp(i, 60);

            Vector3 trajectory_dir0 = Quat.quat_mul_vec(Quat.quat_inv(db.bone_rotations[i][0]),
                Quat.quat_mul_vec(db.bone_rotations[t0][0], new Vector3(0f, 0f, 1f)));
            Vector3 trajectory_dir1 = Quat.quat_mul_vec(Quat.quat_inv(db.bone_rotations[i][0]),
                Quat.quat_mul_vec(db.bone_rotations[t1][0], new Vector3(0f, 0f, 1f)));
            Vector3 trajectory_dir2 = Quat.quat_mul_vec(Quat.quat_inv(db.bone_rotations[i][0]),
                Quat.quat_mul_vec(db.bone_rotations[t2][0], new Vector3(0f, 0f, 1f)));

            db.features[i][offset + 0] = trajectory_dir0.x;
            db.features[i][offset + 1] = trajectory_dir0.z;
            db.features[i][offset + 2] = trajectory_dir1.x;
            db.features[i][offset + 3] = trajectory_dir1.z;
            db.features[i][offset + 4] = trajectory_dir2.x;
            db.features[i][offset + 5] = trajectory_dir2.z;
        }

        normalize_features(db.features, db.features_offset, db.features_scale, offset, 6, weight);

        offset += 6;
    }
    private static void database_build_bounds(ref database db)
    {
        int nbound_sm = ((db.nframes() + BOUND_SM_SIZE - 1) / BOUND_SM_SIZE);
        int nbound_lr = ((db.nframes() + BOUND_LR_SIZE - 1) / BOUND_LR_SIZE);

        db.bound_sm_min = new float[nbound_sm][];
        db.bound_sm_max = new float[nbound_sm][];
        for (int i = 0; i< nbound_sm; i++)
        {
            db.bound_sm_min[i] = new float[db.nfeatures()];
            db.bound_sm_max[i] = new float[db.nfeatures()];
            for(int j=0; j<db.nfeatures(); j++)
            {
                db.bound_sm_min[i][j] = FLT_MAX;
                db.bound_sm_max[i][j] = -FLT_MAX;
            }
        }
        db.bound_lr_min = new float[nbound_lr][];
        db.bound_lr_max = new float[nbound_lr][];
        for (int i = 0; i < nbound_lr; i++)
        {
            db.bound_lr_min[i] = new float[db.nfeatures()];
            db.bound_lr_max[i] = new float[db.nfeatures()];
            for(int j=0; j < db.nfeatures(); j++)
            {
                db.bound_lr_min[i][j] = FLT_MAX;
                db.bound_lr_max[i][j] = -FLT_MAX;
            }
        }

        for(int i=0; i<db.nframes(); i++)
        {
            int i_sm = i / BOUND_SM_SIZE;
            int i_lr = i / BOUND_LR_SIZE;

            for(int j=0; j<db.nfeatures(); j++)
            {
                db.bound_sm_min[i_sm][j] = Mathf.Min(db.bound_sm_min[i_sm][j], db.features[i][j]);
                db.bound_sm_max[i_sm][j] = Mathf.Max(db.bound_sm_max[i_sm][j], db.features[i][j]);
                db.bound_lr_min[i_lr][j] = Mathf.Min(db.bound_lr_min[i_lr][j], db.features[i][j]);
                db.bound_lr_max[i_lr][j] = Mathf.Max(db.bound_lr_max[i_lr][j], db.features[i][j]);
            }
        }
    }
    public static void database_build_matching_features(ref database db, float weight_foot_position, float weight_foot_veloity, 
        float weight_hip_velocity, float weight_trajectory_position, float weight_trajectory_direction, float weight_trajectory_toe_height, float foot_height)
    {
        int nfeatures = 3 + 3 + 3 + 3 + 3 + 6 + 6 + 4 + 4;

        db.features = new float[db.nframes()][];
        for(int i=0; i<db.features.Length; i++)
        {
            db.features[i]=new float[nfeatures];
        }
        db.features_offset = new float[nfeatures];
        db.features_scale = new float[nfeatures];

        int offset = 0;
        compute_bone_position_feature(ref db, ref offset, (int)MotionMatcher.character.Bone_LeftFoot, weight_foot_position);
        compute_bone_position_feature(ref db, ref offset, (int)MotionMatcher.character.Bone_RightFoot, weight_foot_position);
        compute_bone_velocity_feature(ref db, ref offset, (int)MotionMatcher.character.Bone_LeftFoot, weight_foot_veloity);
        compute_bone_velocity_feature(ref db, ref offset, (int)MotionMatcher.character.Bone_RightFoot, weight_foot_veloity);
        compute_bone_velocity_feature(ref db, ref offset, (int)MotionMatcher.character.Bone_Hips, weight_hip_velocity);
        compute_trajectory_position_feature(ref db, ref offset, weight_trajectory_position);
        compute_trajectory_direction_feature(ref db, ref offset, weight_trajectory_direction);
        compute_bone_height_feature(ref db, ref offset, (int)MotionMatcher.character.Bone_LeftToe, foot_height, weight_trajectory_toe_height);
        compute_bone_height_feature(ref db, ref offset, (int)MotionMatcher.character.Bone_RightToe, foot_height, weight_trajectory_toe_height);

        Debug.Assert(offset == nfeatures);

        database_build_bounds(ref db);
    }
    public static void database_save_matching_features(database db, String filename)
    {
        try
        {
            using (FileStream fs = new FileStream(filename, FileMode.CreateNew, FileAccess.Write))
            using (BinaryWriter bw = new BinaryWriter(fs))
            {
                bw.Write(db.features.Length);
                bw.Write(db.features[0].Length);
                foreach (float[] features in db.features)
                {
                    foreach(float feature in features)
                    {
                        bw.Write(feature);
                    }
                }

                bw.Write(db.features_offset.Length);
                foreach (float offset in db.features_offset)
                {
                    bw.Write(offset);
                }

                bw.Write(db.features_scale.Length);
                foreach (float scale in db.features_scale)
                {
                    bw.Write(scale);
                }
            }
        }catch(IOException e)
        {
            Debug.LogException(e);
        }
    }
    #endregion

    #region Readers
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
    #endregion

    #region Loaders
    public static Model Load_net_fromParameters(string filename)
    {
        string path = Path.Combine(Application.streamingAssetsPath, filename);
        using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read))
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
    public static database load_database(string filename)
    {
        string path = Path.Combine(Application.streamingAssetsPath, filename);
        database db = new database();
        using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read))
        using (BinaryReader reader = new BinaryReader(fs))
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

            rows = reader.ReadInt32();
            cols = reader.ReadInt32();
            db.terrain_positions = readVec3_toArray2d(reader, rows, cols / 3);

            rows = reader.ReadInt32();
            cols = reader.ReadInt32();
            db.traj_toe_positions = readVec3_toArray2d(reader, rows, cols / 3);

        }
        return db;
    }

    public static (float[][], float[], float[]) load_features(string filename)
    {
        string path = Path.Combine(Application.streamingAssetsPath, filename);
        float[][] features;
        float[] features_offset;
        float[] features_scale;
        using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read))
        using (BinaryReader reader = new BinaryReader(fs))
        {
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            features = readFloat_toArray2d(reader, rows, cols);

            int count = reader.ReadInt32();
            features_offset = readFloat_toArray(reader, count);

            count = reader.ReadInt32();
            features_scale = readFloat_toArray(reader, count);
        }
        return (features, features_offset, features_scale);
    }

    public static float[][] load_latent(string filename)
    {
        string path = Path.Combine(Application.streamingAssetsPath, filename);
        float[][] latents;
        using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read))
        using (BinaryReader reader = new BinaryReader(fs))
        {
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            latents = readFloat_toArray2d(reader, rows, cols);

        }

        return latents;
    }

    public static character load_character(string filename)
    {
        string path = Path.Combine(Application.streamingAssetsPath, filename);
        character c = new character();
        using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read))
        using (BinaryReader reader = new BinaryReader(fs))
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
            c.bone_weights = readFloat_toArray2d(reader, rows, cols);

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
    public static Mesh gen_mesh_from_character(character c)
    {
        Mesh mesh = new Mesh();

        mesh.vertices = c.positions;
        mesh.uv = c.texcoords;
        mesh.normals = c.normals;

        int[] triangles = new int[c.triangles.Length];
        for(int i=0; i<triangles.Length; i++)
        {
            triangles[i] = c.triangles[i];
        }
        mesh.triangles = triangles;


        mesh.RecalculateBounds();
        mesh.RecalculateTangents();
        mesh.UploadMeshData(false);

        mesh.MarkDynamic();

        return mesh;
    }
    #endregion

    #region Structs
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

        public Vector3[][] terrain_positions;

        public Vector3[][] traj_toe_positions;

        public float[][] bound_sm_min;
        public float[][] bound_sm_max;
        public float[][] bound_lr_min;
        public float[][] bound_lr_max;

        public int nframes() { return bone_positions.Length; }
        public int nbones() { return bone_positions[0].Length; }
        public int nfeatures() { return features[0].Length; }
        public int nranges() { return range_starts.Length; }
        public int ncontacts() { return contact_states[0].Length; }

        public int database_trajectory_index_clamp(int frame, int offset)
        {
            for (int i = 0; i < this.nranges(); i++)
            {
                if (frame >= this.range_starts[i] && frame < this.range_stops[i])
                {
                    return clamp(frame + offset, this.range_starts[i], this.range_stops[i] - 1);
                }
            }

            Debug.Assert(false);

            return -1;
        }


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

        public static void liner_blend_skinning_positions(character c, Pose pose, ref Vector3[] anim_positions)
        {
            for(int i=0; i<anim_positions.Length; i++)
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
        public static void liner_blend_skinning_normals(character c, Pose pose, ref Vector3[] anim_normals)
        {
            for(int i=0; i<anim_normals.Length; i++)
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
    #endregion

    #region FKs
    private static void forward_kinematics(out Vector3 bone_pos, out Vector4 bone_rot,
        Vector3[] bone_positions, Vector4[] bone_rotations, int[] bone_parents, int bone)
    {
        if (bone_parents[bone] != -1)
        {
            Vector3 parent_pos;
            Vector4 parent_rot;

            forward_kinematics(out parent_pos, out parent_rot,
                bone_positions, bone_rotations, bone_parents, bone_parents[bone]);

            bone_pos = Quat.quat_mul_vec(parent_rot, bone_positions[bone]) + parent_pos;
            bone_rot = Quat.quat_mul(parent_rot, bone_rotations[bone]);
        }
        else
        {
            bone_pos = bone_positions[bone];
            bone_rot = bone_rotations[bone];
        }
    }
    private static void forward_kinematics_velocity(out Vector3 bone_pos, out Vector4 bone_rot, out Vector3 bone_vel, out Vector3 bone_angular_vel,
        Vector3[] positions, Vector4[] rotations, Vector3[] velocities, Vector3[] angular_velocities, int[] bone_parents, int bone)
    {
        if (bone_parents[bone] != -1)
        {
            Vector3 parent_pos;
            Vector3 parent_vel;
            Vector4 parent_rot;
            Vector3 parent_ang_vel;

            forward_kinematics_velocity(out parent_pos, out parent_rot, out parent_vel, out parent_ang_vel,
                positions, rotations, velocities, angular_velocities, bone_parents, bone_parents[bone]);

            bone_pos = Quat.quat_mul_vec(parent_rot, positions[bone]) + parent_pos;
            bone_vel = parent_vel + Quat.quat_mul_vec(parent_rot, velocities[bone]) +
                Quat._cross(parent_ang_vel, Quat.quat_mul_vec(parent_rot, positions[bone]));
            bone_rot = Quat.quat_mul(parent_rot, rotations[bone]);
                bone_angular_vel = Quat.quat_mul_vec(parent_rot, angular_velocities[bone]) + parent_ang_vel;
            }
        else
        {
            bone_pos = positions[bone];
            bone_rot = rotations[bone];
            bone_vel = velocities[bone];
            bone_angular_vel = angular_velocities[bone];
        }
    }
    #endregion
    private static int clamp(int x, int min, int max)
    {
        return x > max ? max : x < min ? min : x;
    }
    private static int index(int bone, int vector, int component, int subcomponent, TensorShape shape)
    {
        return bone * shape.height * shape.width * shape.channels +
               vector * shape.width * shape.channels +
               component * shape.channels +
               subcomponent;
    }
    private static float squaref(float x) { return x * x; }
}
