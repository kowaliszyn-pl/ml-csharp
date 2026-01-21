// Neural Networks in C♯
// File name: ModelUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks.Utils;

public static class ModelUtils
{
    public static string GetTypeIdentifier(Type type)
        => type.AssemblyQualifiedName ?? type.FullName ?? type.Name;
}
