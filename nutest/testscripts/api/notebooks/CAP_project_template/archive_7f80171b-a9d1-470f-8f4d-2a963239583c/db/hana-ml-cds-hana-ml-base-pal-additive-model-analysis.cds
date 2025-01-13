namespace hana.ml;

context Fit {
    entity ModelHanaMlConsPalAdditiveModelAnalysis {
    row_index      : Integer;
    model_content  : LargeString;
  }
}
context Predict {
    entity Output0PalAdditiveModelPredict {
    booking_date  : Date;
    yhat          : Double;
    yhat_lower    : Double;
    yhat_upper    : Double;
  }
}