namespace hana.ml;

context Fit {
  @cds.persistence.exists
  entity ModelHanaMlConsPalAutomlFit {
    id        : Integer;
    pipeline  : String(5000);
    scores    : String(5000);
  }
  @cds.persistence.exists
  entity Output1PalAutomlFit {
    row_index      : Integer;
    model_content  : String(5000);
  }
  @cds.persistence.exists
  entity Output2PalAutomlFit {
    stat_name   : String(5000);
    stat_value  : String(5000);
  }
}