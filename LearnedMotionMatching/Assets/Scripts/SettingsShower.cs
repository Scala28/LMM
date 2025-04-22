// ConditionalFieldDrawer.cs
using UnityEditor;
using UnityEngine;

[CustomPropertyDrawer(typeof(ConditionalFieldAttribute))]
public class SettingsShower : PropertyDrawer
{
    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        ConditionalFieldAttribute condAttr = (ConditionalFieldAttribute)attribute;
        SerializedProperty sourceProperty = property.serializedObject.FindProperty(condAttr.conditionFieldName);

        if (sourceProperty != null && sourceProperty.propertyType == SerializedPropertyType.Boolean)
        {
            if (sourceProperty.boolValue == condAttr.value)
            {
                EditorGUI.PropertyField(position, property, label, true);
            }
        }
    }

    public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
    {
        ConditionalFieldAttribute condAttr = (ConditionalFieldAttribute)attribute;
        SerializedProperty sourceProperty = property.serializedObject.FindProperty(condAttr.conditionFieldName);

        if (sourceProperty != null && sourceProperty.propertyType == SerializedPropertyType.Boolean)
        {
            return sourceProperty.boolValue == condAttr.value ? EditorGUI.GetPropertyHeight(property, label, true) : 0f;
        }

        return 0f;
    }
}
public class ConditionalFieldAttribute : PropertyAttribute
{
    public string conditionFieldName;
    public bool value;

    public ConditionalFieldAttribute(string conditionFieldName, bool value)
    {
        this.conditionFieldName = conditionFieldName;
        this.value = value;
    }
}
