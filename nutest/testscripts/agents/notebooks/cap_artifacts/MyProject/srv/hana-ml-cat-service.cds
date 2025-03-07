using hana.ml as hanaml from '../db/hana-ml-cds';

service CatalogService {
    @readonly entity Input0PalAutoarima as projection on hanaml.Fit.Input0PalAutoarima;
    @readonly entity ModelHanaMlConsPalAutoarima as projection on hanaml.Fit.ModelHanaMlConsPalAutoarima;
    @readonly entity Output1PalAutoarima as projection on hanaml.Fit.Output1PalAutoarima;
}