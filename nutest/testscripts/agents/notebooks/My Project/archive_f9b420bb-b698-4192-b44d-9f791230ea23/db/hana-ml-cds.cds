namespace hana.ml;

context Fit {
    entity ModelHanaMlConsPalAutomlFit {
    id        : Integer;
    pipeline  : String(5000);
    scores    : LargeString;
  }
    entity Output1PalAutomlFit {
    row_index      : Integer;
    model_content  : String(5000);
  }
    entity Output2PalAutomlFit {
    stat_name   : String(256);
    stat_value  : LargeString;
  }
}