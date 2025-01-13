using hana.ml as hanaml from '../db/hana-ml-cds';

service CatalogService {
    @readonly entity Input0PalAdditiveModelAnalysis as projection on hanaml.Fit.Input0PalAdditiveModelAnalysis;
    @readonly entity ModelHanaMlConsPalAdditiveModelAnalysis as projection on hanaml.Fit.ModelHanaMlConsPalAdditiveModelAnalysis;
}