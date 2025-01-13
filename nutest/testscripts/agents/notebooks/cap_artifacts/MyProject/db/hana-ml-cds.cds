namespace hana.ml;

context Fit {
    entity Input0PalAdditiveModelAnalysis {
    index  : Timestamp;
    y      : Double;
  }
    entity ModelHanaMlConsPalAdditiveModelAnalysis {
    row_index      : Integer;
    model_content  : LargeString;
  }
}