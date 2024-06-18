using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;


public static class DataManager
{
    // Read .bin file from Resources folder
    public static (int, int, float[]) LoadBin_fromResources(string filename)
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
            float[] data = new float[(memStream.Length - 8) / sizeof(float)];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = reader.ReadSingle();
            }
            return (nframes, ndata, data);
        }
    }
}
