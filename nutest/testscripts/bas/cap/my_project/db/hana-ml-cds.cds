namespace hana.ml;

context Fit {
    entity ModelHanaMlConsPalAutomlFit {
    id        : Integer;
    pipeline  : String(5000);
    scores    : String(5000);
  }
    entity Output1PalAutomlFit {
    row_index      : Integer;
    model_content  : String(5000);
  }
    entity Output2PalAutomlFit {
    stat_name   : String(5000);
    stat_value  : String(5000);
  }
}
context Predict {
    entity Output0PalPipelinePredict {
    id      : Integer;
    scores  : String(5000);
  }
    entity Output1PalPipelinePredict {
    stat_name   : String(5000);
    stat_value  : String(5000);
  }
}