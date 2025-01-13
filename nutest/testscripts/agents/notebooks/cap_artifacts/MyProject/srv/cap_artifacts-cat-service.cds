using cap_artifacts as hanaml from '../db/cap_artifacts-cds';

service CatalogService {
    @readonly entity Input0PalAdditiveModelAnalysis as projection on hanaml.Fit.Input0PalAdditiveModelAnalysis;
    @readonly entity ModelCapArtifactsConsPalAdditiveModelAnalysis as projection on hanaml.Fit.ModelCapArtifactsConsPalAdditiveModelAnalysis;
}