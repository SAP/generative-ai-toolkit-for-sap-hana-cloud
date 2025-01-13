using hana.ml as hanaml from '../db/hana-ml-cds-hana-ml-base-pal-additive-model-analysis';

service CatalogService {
    @readonly entity ModelHanaMlConsPalAdditiveModelAnalysis as projection on hanaml.Fit.ModelHanaMlConsPalAdditiveModelAnalysis;
}