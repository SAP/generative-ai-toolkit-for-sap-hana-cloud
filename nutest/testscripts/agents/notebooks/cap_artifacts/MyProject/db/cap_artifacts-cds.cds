namespace cap_artifacts;

context Fit {
    entity Input0PalAdditiveModelAnalysis {
    index  : Timestamp;
    y      : Double;
  }
    entity ModelCapArtifactsConsPalAdditiveModelAnalysis {
    row_index      : Integer;
    model_content  : LargeString;
  }
}