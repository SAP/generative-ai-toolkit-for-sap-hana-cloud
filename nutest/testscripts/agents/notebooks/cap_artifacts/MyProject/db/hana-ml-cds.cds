namespace hana.ml;

context Fit {
    entity Input0PalAutoarima {
    id     : Integer;
    sales  : Double;
  }
    entity ModelHanaMlConsPalAutoarima {
    key    : String(100);
    value  : String(5000);
  }
    entity Output1PalAutoarima {
    id         : Integer;
    fitted     : Double;
    residuals  : Double;
  }
}