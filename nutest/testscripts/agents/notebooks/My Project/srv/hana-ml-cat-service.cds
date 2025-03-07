using hana.ml as hanaml from '../db/hana-ml-cds';

service CatalogService {
    @readonly entity ModelHanaMlConsPalAutomlFit as projection on hanaml.Fit.ModelHanaMlConsPalAutomlFit;
    @readonly entity Output1PalAutomlFit as projection on hanaml.Fit.Output1PalAutomlFit;
    @readonly entity Output2PalAutomlFit as projection on hanaml.Fit.Output2PalAutomlFit;
}